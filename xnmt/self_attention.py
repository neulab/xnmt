import numpy as np
import dynet as dy
from math import sqrt

from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.param_collection import ParamManager
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt.transducer import SeqTransducer

class MultiHeadAttentionSeqTransducer(SeqTransducer, Serializable):
  """
  This implements the Multi-headed attention layer of "Attention is All You Need":
  https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

  Args:

  """
  yaml_tag = '!MultiHeadAttentionSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               num_heads=8,
               yaml_path=None):
    assert(input_dim % num_heads == 0)

    param_collection = ParamManager.my_params(self)

    self.input_dim = input_dim
    self.num_heads = num_heads
    self.head_dim = input_dim // num_heads

    # self.Wq, self.Wk, self.Wv, self.Wo = [param_collection.add_parameters(dim=(input_dim, input_dim), init=param_init) for _ in range(4)]
    # self.bq, self.bk, self.bv, self.bo = [param_collection.add_parameters(dim=(input_dim), init=bias_init) for _ in range(4)]
    self.Wq, self.Wk, self.Wv, self.Wo = [param_collection.add_parameters(dim=(input_dim, input_dim)) for _ in range(4)]
    self.bq, self.bk, self.bv, self.bo = [param_collection.add_parameters(dim=(1, input_dim)) for _ in range(4)]

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def __call__(self, expr_seq):
    """
    transduce the sequence

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """

    # Start with a [(length, model_size) x batch] tensor
    x = expr_seq.as_transposed_tensor()
    x_len = x.dim()[0][0]
    x_batch = x.dim()[1]
    # Get the query key and value vectors
    # TODO: do we need bias broadcasting in DyNet?
    # q = dy.affine_transform([self.bq, x, self.Wq])
    # k = dy.affine_transform([self.bk, x, self.Wk])
    # v = dy.affine_transform([self.bv, x, self.Wv])
    q = self.bq + x * self.Wq
    k = self.bk + x * self.Wk
    v = self.bv + x * self.Wv
    
    # Split to batches [(length, head_dim) x batch * num_heads] tensor
    q, k, v = [dy.reshape(x, (x_len, self.head_dim), batch_size=x_batch * self.num_heads) for x in (q,k,v)]

    # Do scaled dot product [(length, length) x batch * num_heads], rows are queries, columns are keys
    attn_score = q * dy.transpose(k) / sqrt(self.head_dim)
    if expr_seq.mask != None:
      mask = dy.inputTensor(expr_seq.mask * -1e10, batched=True)
      attn_score = attn_score + mask
    attn_prob = dy.softmax(attn_score, d=1)
    # Reduce using attention and resize to match [(length, model_size) x batch]
    o = dy.reshape(attn_prob * v, (x_len, self.input_dim), batch_size=x_batch)
    # Final transformation
    # o = dy.affine_transform([self.bo, attn_prob * v, self.Wo])
    o = self.bo + o * self.Wo

    return ExpressionSequence(expr_transposed_tensor=o, mask=expr_seq.mask)


