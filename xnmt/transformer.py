import numpy as np
import dynet as dy
import itertools
from xnmt.linear import Linear
from xnmt.mlp import MLP
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.param_init import LeCunUniformInitializer
from xnmt.param_collection import ParamManager

MIN_VALUE = -10000

class LayerNorm(object):
  def __init__(self, dy_model, d_hid):
    self.p_g = dy_model.add_parameters(dim=d_hid)
    self.p_b = dy_model.add_parameters(dim=d_hid)

  def __call__(self, input_expr):
    g = dy.parameter(self.p_g)
    b = dy.parameter(self.p_b)

    (_, seq_len), batch_size = input_expr.dim()
    input = TimeDistributed()(input_expr)
    output = dy.layer_norm(input, g, b)
    return ReverseTimeDistributed()(output, seq_len, batch_size)


class MultiHeadAttention(object):
  """ Multi Head Attention Layer for Sentence Blocks
  """
  def __init__(self, dy_model, n_units, h=1, attn_dropout=False):
    self.W_Q = dy_model.add_parameters((n_units, n_units))
    self.W_K = dy_model.add_parameters((n_units, n_units))
    self.W_V = dy_model.add_parameters((n_units, n_units))
    self.finishing_linear_layer = dy_model.add_parameters((n_units, n_units))
    self.h = h
    self.scale_score = 1. / (n_units / h) ** 0.5
    self.attn_dropout = attn_dropout

  def split_batch(self, X, h):
    (n_rows, _), batch = X.dim()
    l = range(batch)
    steps = batch // h
    output = []
    for i in range(0, batch, steps):
      indexes = l[i:i + steps]
      output.append(dy.pick_batch_elems(X, indexes))
    return output

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, x, z=None, mask=None):
    h = self.h
    if z == None: z = x
    Q = self.W_Q * x
    K = self.W_K * z
    V = self.W_V * z

    (n_units, n_querys), batch = Q.dim()
    (_, n_keys), _ = K.dim()

    assert(n_units % h == 0)

    batch_Q = dy.reshape(dy.transpose(Q), (n_querys, n_units//h), batch_size=batch*h)
    batch_K = dy.reshape(dy.transpose(K), (n_keys, n_units//h), batch_size=batch*h)
    batch_V = dy.reshape(dy.transpose(V), (n_keys, n_units//h), batch_size=batch*h)
    
    batch_A = (batch_Q * dy.transpose(batch_K)) * self.scale_score

    if not isinstance(mask, type(None)):
      mask = dy.inputTensor(mask, batched=True)
      batch_A = batch_A + (1 - mask)*MIN_VALUE

    sent_len = batch_A.dim()[0][0]
    if sent_len == 1:
        batch_A = dy.softmax(batch_A)
    else:
        batch_A = dy.softmax(batch_A, d=1)

    if self.attn_dropout:
      if self.dropout != 0.0:
        batch_A = dy.dropout(batch_A, self.dropout)

    batch_C = dy.transpose(batch_A * batch_V)
    assert (batch_C.dim() == ((n_units // h, n_querys), batch * h))

    C = dy.concatenate(self.split_batch(batch_C, h), d=0)
    assert (C.dim() == ((n_units, n_querys), batch))
    C = self.finishing_linear_layer * C
    return C

class EncoderLayer(object):
  def __init__(self, dy_model, n_units, h=1, attn_dropout=False, layer_norm=False):
    self.self_attention = MultiHeadAttention(dy_model, n_units, h, attn_dropout=attn_dropout)
    self.feed_forward = MLP(input_dim=n_units, hidden_dim=n_units*4, output_dim=n_units, activation='relu')
    self.layer_norm = layer_norm
    if self.layer_norm:
      self.ln_1 = LayerNorm(dy_model, n_units)
      self.ln_2 = LayerNorm(dy_model, n_units)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, xx_mask):
    self.self_attention.set_dropout(self.dropout)
    sub = self.self_attention(e, mask=xx_mask)
    if self.dropout != 0.0:
      sub = dy.dropout(sub, self.dropout)
    e = e + sub
    if self.layer_norm:
      e = self.ln_1(e)

    sub = self.feed_forward(e)
    if self.dropout != 0.0:
      sub = dy.dropout(sub, self.dropout)
    e = e + sub
    if self.layer_norm:
      e = self.ln_2(e)
    return e


class DecoderLayer(object):
  def __init__(self, dy_model, n_units, h=1, attn_dropout=False, layer_norm=False):
    self.self_attention = MultiHeadAttention(dy_model, n_units, h, attn_dropout=attn_dropout)
    self.source_attention = MultiHeadAttention(dy_model, n_units, h, attn_dropout=attn_dropout)
    self.feed_forward = MLP(input_dim=n_units, hidden_dim=n_units*4, output_dim=n_units, activation='relu')
    self.layer_norm = layer_norm
    if self.layer_norm:
      self.ln_1 = LayerNorm(dy_model, n_units)
      self.ln_2 = LayerNorm(dy_model, n_units)
      self.ln_3 = LayerNorm(dy_model, n_units)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, s, xy_mask, yy_mask):
    self.self_attention.set_dropout(self.dropout)
    sub = self.self_attention(e, mask=yy_mask)
    if self.dropout != 0.0:
      sub = dy.dropout(sub, self.dropout)
    e = e + sub
    if self.layer_norm:
      e = self.ln_1(e)

    self.source_attention.set_dropout(self.dropout)
    sub = self.source_attention(e, s, mask=xy_mask)
    if self.dropout != 0.0:
      sub = dy.dropout(sub, self.dropout)
    e = e + sub
    if self.layer_norm:
      e = self.ln_2(e)

    sub = self.feed_forward(e)
    if self.dropout != 0.0:
      sub = dy.dropout(sub, self.dropout)
    e = e + sub
    if self.layer_norm:
      e = self.ln_3(e)
    return e


class TransformerEncoder(Serializable):
  yaml_tag = '!TransformerEncoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, layers=1, input_dim=512, h=1,
               dropout=0.0, attn_dropout=False, layer_norm=False, **kwargs):
    dy_model = ParamManager.my_params(self)
    self.layer_names = []
    for i in range(1, layers + 1):
      name = 'l{}'.format(i)
      layer = EncoderLayer(dy_model, input_dim, h, attn_dropout, layer_norm)
      self.layer_names.append((name, layer))

    self.dropout_val = dropout

  @handle_xnmt_event
  def on_set_train(self, val):
    self.set_dropout(self.dropout_val if val else 0.0)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, xx_mask):
    if self.dropout != 0.0:
      e = dy.dropout(e, self.dropout)  # Word Embedding Dropout

    for name, layer in self.layer_names:
      layer.set_dropout(self.dropout)
      e = layer(e, xx_mask)
    return e


class TransformerDecoder(Serializable):
  yaml_tag = '!TransformerDecoder'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, layers=1, input_dim=512, h=1,
               dropout=0.0, attn_dropout=False, layer_norm=False,
               vocab_size = None, vocab = None,
               trg_reader = Ref("model.trg_reader")):
    dy_model = ParamManager.my_params(self)
    self.layer_names = []
    for i in range(1, layers + 1):
      name = 'l{}'.format(i)
      layer = DecoderLayer(dy_model, input_dim, h, attn_dropout, layer_norm)
      self.layer_names.append((name, layer))

    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, trg_reader)
    self.output_affine = Linear(input_dim, self.vocab_size, dy_model, param_init=LeCunUniformInitializer(), bias_init=LeCunUniformInitializer())
    self.dropout_val = dropout

  def choose_vocab_size(self, vocab_size, vocab, trg_reader):
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path
    """
    if vocab_size != None:
      return vocab_size
    elif vocab != None:
      return len(vocab)
    elif trg_reader == None or trg_reader.vocab == None:
      raise ValueError("Could not determine trg_embedder's size. Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.set_dropout(self.dropout_val if val else 0.0)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, source, xy_mask, yy_mask):
    if self.dropout != 0.0:
      e = dy.dropout(e, self.dropout)  # Word Embedding Dropout
    for name, layer in self.layer_names:
      layer.set_dropout(self.dropout)
      e = layer(e, source, xy_mask, yy_mask)
    return e

  def loss_from_logits(self, logits, trg_words):
    (vocab, sent_len), batches = logits.dim()
    logits = dy.reshape(logits, (vocab,), batch_size=sent_len * batches)
    flattened_trg_words = itertools.chain(*trg_words)
    loss = dy.pickneglogsoftmax_batch(logits, flattened_trg_words)
    return loss


