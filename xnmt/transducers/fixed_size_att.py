from typing import Optional, List
import numbers

import dynet as dy
import numpy as np

from xnmt import expression_seqs, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init, Ref, bare

class FixedSizeAttSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  A fixed-size attention-based representation of a sequence.

  This implements the basic fixed-size memory model according to Britz et. al 2017: ﻿Efficient Attention using a
  Fixed-Size Memory Representation; https://arxiv.org/abs/1707.00110

  Args:
    hidden_dim: hidden dimension of inputs and outputs
    output_len: fixed-size length of the output
    pos_enc_max: if given, use positional encodings, assuming the number passed here as the maximum possible input
                 sequence length
    param_init: parameter initializer
  """
  yaml_tag = "!FixedSizeAttSeqTransducer"

  @serializable_init
  def __init__(self,
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_len: numbers.Integral = 32,
               pos_enc_max: Optional[numbers.Integral] = None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))) \
          -> None:
    subcol = param_collections.ParamManager.my_params(self)
    self.output_len = output_len
    self.W = subcol.add_parameters(dim=(hidden_dim, output_len),
                                   init=param_init.initializer((hidden_dim, output_len)))
    self.pos_enc_max = pos_enc_max
    if self.pos_enc_max:
      self.pos_enc = np.zeros((self.pos_enc_max, self.output_len))
      for k in range(self.output_len):
        for s in range(self.pos_enc_max):
          self.pos_enc[s, k] = (1.0 - k / self.output_len) * (
                  1.0 - s / self.pos_enc_max) + k / self.output_len * s / self.pos_enc_max

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    raise NotImplementedError('FixedSizeAttSeqTransducer.get_final_states() not implemented')

  def transduce(self, x: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    x_T = x.as_transposed_tensor()
    scores = x_T * dy.parameter(self.W)
    if x.mask is not None:
      scores = x.mask.add_to_tensor_expr(scores, multiplicator=-100.0, time_first=True)
    if self.pos_enc_max:
      seq_len = x_T.dim()[0][0]
      pos_enc = self.pos_enc[:seq_len,:]
      scores = dy.cmult(scores, dy.inputTensor(pos_enc))
    attention = dy.softmax(scores)
    output_expr = x.as_tensor() * attention
    return expression_seqs.ExpressionSequence(expr_tensor=output_expr, mask=None)
