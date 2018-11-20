import numpy as np
import dynet as dy
from math import sqrt
from typing import List
import numbers

from xnmt import events, expression_seqs, param_collections, param_initializers
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt.transducers import base as transducers

class MultiHeadAttentionSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements the Multi-headed attention layer of "Attention is All You Need":
  https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

  Args:
    input_dim: size of inputs
    dropout: dropout to apply to attention matrix
    param_init: how to initialize param matrices
    bias_init: how to initialize bias params
    num_heads: number of attention heads
  """
  yaml_tag = '!MultiHeadAttentionSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               num_heads: numbers.Integral = 8):
    assert(input_dim % num_heads == 0)

    self.dropout = dropout

    param_collection = param_collections.ParamManager.my_params(self)

    self.input_dim = input_dim
    self.num_heads = num_heads
    self.head_dim = input_dim // num_heads

    self.pWq, self.pWk, self.pWv, self.pWo = [param_collection.add_parameters(dim=(input_dim, input_dim), init=param_init.initializer((input_dim, input_dim))) for _ in range(4)]
    self.pbq, self.pbk, self.pbv, self.pbo = [param_collection.add_parameters(dim=(1, input_dim), init=bias_init.initializer((1, input_dim,))) for _ in range(4)]

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def transduce(self, expr_seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    """
    transduce the sequence

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """

    Wq, Wk, Wv, Wo = [dy.parameter(x) for x in (self.pWq, self.pWk, self.pWv, self.pWo)]
    bq, bk, bv, bo = [dy.parameter(x) for x in (self.pbq, self.pbk, self.pbv, self.pbo)]

    # Start with a [(length, model_size) x batch] tensor
    x = expr_seq.as_transposed_tensor()
    x_len = x.dim()[0][0]
    x_batch = x.dim()[1]
    # Get the query key and value vectors
    # TODO: do we need bias broadcasting in DyNet?
    # q = dy.affine_transform([bq, x, Wq])
    # k = dy.affine_transform([bk, x, Wk])
    # v = dy.affine_transform([bv, x, Wv])
    q = bq + x * Wq
    k = bk + x * Wk
    v = bv + x * Wv
    
    # Split to batches [(length, head_dim) x batch * num_heads] tensor
    q, k, v = [dy.reshape(x, (x_len, self.head_dim), batch_size=x_batch * self.num_heads) for x in (q,k,v)]

    # Do scaled dot product [(length, length) x batch * num_heads], rows are queries, columns are keys
    attn_score = q * dy.transpose(k) / sqrt(self.head_dim)
    if expr_seq.mask is not None:
      mask = dy.inputTensor(np.repeat(expr_seq.mask.np_arr, self.num_heads, axis=0).transpose(), batched=True) * -1e10
      attn_score = attn_score + mask
    attn_prob = dy.softmax(attn_score, d=1)
    if self.train and self.dropout > 0.0:
      attn_prob = dy.dropout(attn_prob, self.dropout)
    # Reduce using attention and resize to match [(length, model_size) x batch]
    o = dy.reshape(attn_prob * v, (x_len, self.input_dim), batch_size=x_batch)
    # Final transformation
    # o = dy.affine_transform([bo, attn_prob * v, Wo])
    o = bo + o * Wo

    expr_seq = expression_seqs.ExpressionSequence(expr_transposed_tensor=o, mask=expr_seq.mask)

    self._final_states = [transducers.FinalTransducerState(expr_seq[-1], None)]

    return expr_seq


