import dynet as dy
import math
import numpy as np
from xnmt.expression_sequence import ExpressionSequence
from xnmt.nn import *

MAX_SIZE = 5000
MIN_VAL = -10000   # This value is close to NEG INFINITY


class MultiHeadedAttention(object):
  def __init__(self, head_count, model_dim, model):
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count

    # Linear Projection of keys
    self.linear_keys = Linear(model_dim, head_count * self.dim_per_head, model)

    # Linear Projection of values
    self.linear_values = Linear(model_dim, head_count * self.dim_per_head, model)

    # Linear Projection of query
    self.linear_query = Linear(model_dim, head_count * self.dim_per_head, model)

    # Layer Norm Module
    self.layer_norm = LayerNorm(model_dim, model)

  def __call__(self, key, value, query, mask, p):
    residual = TimeDistributed()(query)
    batch_size = key[0].dim()[1]

    def shape_projection(x):
      total_words = x.dim()[1]
      seq_len = total_words / batch_size
      temp = dy.reshape(x, (self.model_dim, seq_len), batch_size=batch_size)
      temp = dy.transpose(temp)
      return dy.reshape(temp, (seq_len, self.dim_per_head), batch_size=batch_size * self.head_count)

    # Concatenate all the words together for doing vectorized affine transform
    key_up = shape_projection(self.linear_keys(TimeDistributed()(key)))
    value_up = shape_projection(self.linear_values(TimeDistributed()(value)))
    query_up = shape_projection(self.linear_query(TimeDistributed()(query)))

    scaled = query_up * dy.transpose(key_up)
    scaled = scaled / math.sqrt(self.dim_per_head)

    # Apply Mask here
    if mask is not None:
      _, l1, l2, b = mask.shape
      assert(b == batch_size)
      # Following 3 operations are essential to convert a numpy matrix of dimensions mask.shape
      # to the dimensions of scaled tensor in correct way

      # m1 = np.broadcast_to(mask.T, (self.head_count, l, l, batch_size))
      m2 = np.moveaxis(mask, [0, 1, 2], [3, 0, 1])
      m3 = (m2.reshape(l1, l2, -1) * MIN_VAL) + 1  # Convert all 0's to 1's and 0's to MIN_VAL+1
      new_mask = dy.inputTensor(m3, batched=True)
      scaled = dy.cmult(scaled, new_mask)

    # Computing Softmax here. Doing double transpose here, as softmax in dynet is applied to each column
    # May be Optimized ? // Dynet Tricks ??
    attn = dy.transpose(dy.softmax(dy.transpose(scaled)))

    # Applying dropout to attention
    drop_attn = dy.dropout(attn, p)

    # Computing weighted attention score
    attn_prod = drop_attn * value_up

    # Reshaping the attn_prod to input query dimensions
    temp = dy.reshape(attn_prod, (len(query), self.dim_per_head * self.head_count), batch_size=batch_size)
    temp = dy.transpose(temp)
    out = dy.reshape(temp, (self.model_dim,), batch_size=batch_size*len(query))

    # Adding dropout and layer normalization
    res = dy.dropout(out, p) + residual
    ret = self.layer_norm(res)
    return ret


def expr_to_sequence(expr_, seq_len, batch_size):
  out_list = []
  for i in range(seq_len):
    indexes = map(lambda x: x + i, range(0, seq_len * batch_size, seq_len))
    out_list.append(dy.pick_batch_elems(expr_, indexes))
  return out_list


class TransformerEncoderLayer(object):
  def __init__(self, size, rnn_size, model, head_count=8, hidden_size=2048):
    self.self_attn = MultiHeadedAttention(head_count, size, model)  # Self Attention
    self.feed_forward = PositionwiseFeedForward(size, hidden_size, model)  # Feed Forward
    self.head_count = head_count

  def set_dropout(self, dropout):
    self.dropout = dropout

  def transduce(self, input):
    seq_len = len(input)
    batch_size = input[0].dim()[1]

    m_src = None
    if input.mask is not None:
        m_src = np.broadcast_to(input.mask.T, (self.head_count, seq_len, seq_len, batch_size))

    mid = self.self_attn(input, input, input, mask=m_src, p=self.dropout)
    out = self.feed_forward(mid, p=self.dropout)

    assert (np.isnan(out.npvalue()).any() == False)  # Check for Nan
    out_list = expr_to_sequence(out, seq_len, batch_size)
    return out_list


class TransformerDecoderLayer(object):
  def __init__(self, size, rnn_size, model, head_count=8, hidden_size=2048):
    self.self_attn = MultiHeadedAttention(head_count, size, model)  # Self Attention
    self.context_attn = MultiHeadedAttention(head_count, size, model)  # Context Attention
    self.feed_forward = PositionwiseFeedForward(size, hidden_size, model)  # Feed Forward

    self.mask = self._get_attn_subsequent_mask(MAX_SIZE)  # Decoder Attention Mask
    self.head_count = head_count

  def set_dropout(self, dropout):
    self.dropout = dropout

  def transduce(self, context, input, src_mask, trg_mask):
    seq_len = len(input)
    model_dim = input[0].dim()[0][0]
    batch_size = input[0].dim()[1]

    dec_mask = None
    if trg_mask is not None:
      # In this, we need to construct the mask in a special way such that word at time step 't' does not see future words
      m_trg = np.broadcast_to(trg_mask.T, (self.head_count, seq_len, seq_len, batch_size))
      tmp = np.broadcast_to(self.mask[:seq_len, :seq_len], (self.head_count, seq_len, seq_len))
      tmp = np.expand_dims(tmp, 3) + m_trg
      dec_mask = (tmp > 0).astype(np.float64)

    query = self.self_attn(input, input, input, mask=dec_mask, p=self.dropout)
    assert (np.isnan(query.npvalue()).any() == False)  # Check for Nan

    query_list = expr_to_sequence(query, seq_len, batch_size)
    query = ExpressionSequence(query_list)

    m_src = None
    if src_mask is not None:
        m_src = np.broadcast_to(src_mask.T, (self.head_count, seq_len, len(context), batch_size))

    mid = self.context_attn(context, context, query, mask=m_src, p=self.dropout)
    out = self.feed_forward(mid, p=self.dropout)

    # Check for Nan
    assert (np.isnan(out.npvalue()).any() == False)
    out_list = expr_to_sequence(out, seq_len, batch_size)
    return out_list, out

  def _get_attn_subsequent_mask(self, size):
      """
      Get an attention mask to avoid using the subsequent info.
      """
      attn_shape = (size, size)
      subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
      return subsequent_mask

