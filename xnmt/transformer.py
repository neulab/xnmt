# encoding: utf-8
from __future__ import division, generators

import numpy as np
import dynet as dy
from xnmt.initializer import LeCunUniform as linear_init
from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event

MIN_VALUE = -10000


class TimeDistributed(object):
  def __call__(self, input):
    (model_dim, seq_len), batch_size = input.dim()
    total_words = seq_len * batch_size
    return dy.reshape(input, (model_dim,), batch_size=total_words)


class ReverseTimeDistributed(object):
  def __call__(self, input, seq_len, batch_size):
    (model_dim,), total_words = input.dim()
    assert (seq_len * batch_size == total_words)
    return dy.reshape(input, (model_dim, seq_len), batch_size=batch_size)


class Linear(object):
  def __init__(self, dy_model, input_dim, output_dim):
    self.W1 = dy_model.add_parameters((output_dim, input_dim), init=linear_init(input_dim))
    self.b1 = dy_model.add_parameters(output_dim, init=linear_init(output_dim))

  def __call__(self, input_expr, reconstruct_shape=True, timedistributed=False):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)

    if not timedistributed:
        input = TimeDistributed()(input_expr)
    else:
        input = input_expr
    output = dy.affine_transform([b1, W1, input])

    if not reconstruct_shape:
        return output
    (_, seq_len), batch_size = input_expr.dim()
    return ReverseTimeDistributed()(output, seq_len, batch_size)


class Linear_nobias(object):
  def __init__(self, dy_model, input_dim, output_dim):
    self.W1 = dy_model.add_parameters((output_dim, input_dim), init=linear_init(input_dim))
    self.output_dim = output_dim

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    b1 = dy.zeros(self.output_dim)
    (_, seq_len), batch_size = input_expr.dim()

    output = dy.affine_transform([b1, W1, input_expr])

    if seq_len == 1: # This is helpful when sequence length is 1, especially during decoding
        output = ReverseTimeDistributed()(output, seq_len, batch_size)
    return output


class LayerNorm(object):
  def __init__(self, dy_model, d_hid):
    self.p_g = dy_model.add_parameters(dim=d_hid)
    self.p_b = dy_model.add_parameters(dim=d_hid)

  def __call__(self, input_expr):
    # g = dy.parameter(self.p_g)
    # b = dy.parameter(self.p_b)
    #
    # (_, seq_len), batch_size = input_expr.dim()
    # input = TimeDistributed()(input_expr)
    # output = dy.layer_norm(input, g, b)
    # return ReverseTimeDistributed()(output, seq_len, batch_size)
    return input_expr


def split_rows(X, h):
  (n_rows, _), batch = X.dim()
  l = range(n_rows)
  steps = n_rows // h
  output = []
  for i in range(0, n_rows, steps):
    output.append(dy.pickrange(X, i, i + steps))
  return output


def split_batch(X, h):
  (n_rows, _), batch = X.dim()
  l = range(batch)
  steps = batch // h
  output = []
  for i in range(0, batch, steps):
    indexes = l[i:i + steps]
    output.append(dy.pick_batch_elems(X, indexes))
  return output


class MultiHeadAttention(object):
  """ Multi Head Attention Layer for Sentence Blocks
  For batch computation efficiency, dot product to calculate query-key
  scores is performed all heads together.
  """
  def __init__(self, dy_model, n_units, h=1, self_attention=True, attn_dropout=False):
    self.W_Q = Linear_nobias(dy_model, n_units, n_units)
    self.W_K = Linear_nobias(dy_model, n_units, n_units)
    self.W_V = Linear_nobias(dy_model, n_units, n_units)
    self.finishing_linear_layer = Linear_nobias(dy_model, n_units, n_units)
    self.h = h
    self.scale_score = 1. / (n_units // h) ** 0.5
    self.is_self_attention = self_attention
    self.attn_dropout = attn_dropout

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, x, z=None, mask=None):
    h = self.h
    if self.is_self_attention:
      Q = self.W_Q(x)
      K = self.W_K(x)
      V = self.W_V(x)
    else:
      Q = self.W_Q(x)
      K = self.W_K(z)
      V = self.W_V(z)

    (n_units, n_querys), batch = Q.dim()
    (_, n_keys), _ = K.dim()

    # Calculate Attention Scores with Mask for Zero-padded Areas
    # Perform Multi-head Attention using pseudo batching
    # all together at once for efficiency

    batch_Q = dy.concatenate_to_batch(split_rows(Q, h))
    batch_K = dy.concatenate_to_batch(split_rows(K, h))
    batch_V = dy.concatenate_to_batch(split_rows(V, h))

    assert(batch_Q.dim() == (n_units // h, n_querys), batch * h)
    assert(batch_K.dim() == (n_units // h, n_keys), batch * h)
    assert(batch_V.dim() == (n_units // h, n_keys), batch * h)

    mask = np.concatenate([mask] * h, axis=0)
    mask = dy.inputTensor(np.moveaxis(mask, [1, 0, 2], [0, 2, 1]), batched=True)
    batch_A = (dy.transpose(batch_Q) * batch_K) * self.scale_score
    batch_A = dy.cmult(batch_A, mask) + (1 - mask)*MIN_VALUE

    sent_len = batch_A.dim()[0][0]
    if sent_len == 1:
        batch_A = dy.softmax(batch_A)
    else:
        batch_A = dy.softmax(batch_A, d=1)

    batch_A = dy.cmult(batch_A, mask)
    assert (batch_A.dim() == ((n_querys, n_keys), batch * h))

    if self.attn_dropout:
      batch_A = dy.dropout(batch_A, self.dropout)

    batch_C = dy.transpose(batch_A * dy.transpose(batch_V))
    assert (batch_C.dim() == ((n_units // h, n_querys), batch * h))

    C = dy.concatenate(split_batch(batch_C, h), d=0)
    assert (C.dim() == ((n_units, n_querys), batch))
    C = self.finishing_linear_layer(C)
    return C


class FeedForwardLayer(object):
  def __init__(self, dy_model, n_units):
    n_inner_units = n_units * 4
    self.W_1 = Linear(dy_model, n_units, n_inner_units)
    self.W_2 = Linear(dy_model, n_inner_units, n_units)
    self.act = dy.rectify

  def __call__(self, e):
    e = self.W_1(e, reconstruct_shape=False, timedistributed=True)
    e = self.act(e)
    e = self.W_2(e, reconstruct_shape=False, timedistributed=True)
    return e


class EncoderLayer(object):
  def __init__(self, dy_model, n_units, h=1, attn_dropout=False):
    self.self_attention = MultiHeadAttention(dy_model, n_units, h, attn_dropout=attn_dropout)
    self.feed_forward = FeedForwardLayer(dy_model, n_units)
    self.ln_1 = LayerNorm(dy_model, n_units)
    self.ln_2 = LayerNorm(dy_model, n_units)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, xx_mask):
    self.self_attention.set_dropout(self.dropout)
    sub = self.self_attention(e, e, xx_mask)
    e = e + dy.dropout(sub, self.dropout)
    e = self.ln_1(e)

    sub = self.feed_forward(e)
    e = e + dy.dropout(sub, self.dropout)
    e = self.ln_2(e)
    return e


class DecoderLayer(object):
  def __init__(self, dy_model, n_units, h=1, attn_dropout=False):
    self.self_attention = MultiHeadAttention(dy_model, n_units, h,
                                             attn_dropout=attn_dropout)
    self.source_attention = MultiHeadAttention(dy_model, n_units, h,
                                               self_attention=False,
                                               attn_dropout=attn_dropout)
    self.feed_forward = FeedForwardLayer(dy_model, n_units)
    self.ln_1 = LayerNorm(dy_model, n_units)
    self.ln_2 = LayerNorm(dy_model, n_units)
    self.ln_3 = LayerNorm(dy_model, n_units)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, s, xy_mask, yy_mask):
    self.self_attention.set_dropout(self.dropout)
    sub = self.self_attention(e, e, yy_mask)
    e = e + dy.dropout(sub, self.dropout)
    e = self.ln_1(e)

    self.source_attention.set_dropout(self.dropout)
    sub = self.source_attention(e, s, xy_mask)
    e = e + dy.dropout(sub, self.dropout)
    e = self.ln_2(e)

    sub = self.feed_forward(e)
    e = e + dy.dropout(sub, self.dropout)
    e = self.ln_3(e)
    return e


class TransformerEncoder(Serializable):
  yaml_tag = u'!TransformerEncoder'

  def __init__(self, yaml_context, layers=1, input_dim=512, h=1,
               dropout=0.0, attn_dropout=False, **kwargs):
    register_handler(self)
    dy_model = yaml_context.dynet_param_collection.param_col
    input_dim = input_dim or yaml_context.default_layer_dim
    self.layer_names = []
    for i in range(1, layers + 1):
      name = 'l{}'.format(i)
      layer = EncoderLayer(dy_model, input_dim, h, attn_dropout)
      self.layer_names.append((name, layer))

    self.dropout_val = dropout or yaml_context.dropout

  @handle_xnmt_event
  def on_set_train(self, val):
    self.set_dropout(self.dropout_val if val else 0.0)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, xx_mask):
    e = dy.dropout(e, self.dropout) # Word Embedding Dropout
    for name, layer in self.layer_names:
      layer.set_dropout(self.dropout)
      e = layer(e, xx_mask)
    return e


class TransformerDecoder(Serializable):
  yaml_tag = u'!TransformerDecoder'

  def __init__(self, yaml_context, vocab_size, layers=1, input_dim=512, h=1,
               label_smoothing=0.0, dropout=0.0, attn_dropout=False, **kwargs):
    register_handler(self)
    dy_model = yaml_context.dynet_param_collection.param_col
    input_dim = input_dim or yaml_context.default_layer_dim
    self.layer_names = []
    for i in range(1, layers + 1):
      name = 'l{}'.format(i)
      layer = DecoderLayer(dy_model, input_dim, h, attn_dropout)
      self.layer_names.append((name, layer))

    self.output_affine = Linear(dy_model, input_dim, vocab_size)
    self.dropout_val = dropout or yaml_context.dropout

  @handle_xnmt_event
  def on_set_train(self, val):
    self.set_dropout(self.dropout_val if val else 0.0)

  def set_dropout(self, dropout):
    self.dropout = dropout

  def __call__(self, e, source, xy_mask, yy_mask):
    e = dy.dropout(e, self.dropout)  # Word Embedding Dropout
    for name, layer in self.layer_names:
      layer.set_dropout(self.dropout)
      e = layer(e, source, xy_mask, yy_mask)
    return e

  def output_and_loss(self, h_block, concat_t_block):
    concat_logit_block = self.output_affine(h_block, reconstruct_shape=False)
    bool_array = concat_t_block != 0
    indexes = np.argwhere(bool_array).ravel()
    concat_logit_block = dy.pick_batch_elems(concat_logit_block, indexes)
    concat_t_block = concat_t_block[bool_array]
    loss = dy.pickneglogsoftmax_batch(concat_logit_block, concat_t_block)
    return loss

  def output(self, h_block):
    concat_logit_block = self.output_affine(h_block, reconstruct_shape=False, timedistributed=True)
    return concat_logit_block


