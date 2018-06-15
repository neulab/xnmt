from typing import Optional

import numpy as np
import dynet as dy

from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt import expression_sequence, param_collection, transducer
import xnmt.param_init


class FixedSizeAttSeqTransducer(transducer.SeqTransducer, Serializable):
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
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_len: int = 32,
               pos_enc_max: Optional[int] = None,
               param_init: xnmt.param_init.ParamInitializer = Ref("exp_global.param_init",
                                                                  default=bare(xnmt.param_init.GlorotInitializer))):
    subcol = param_collection.ParamManager.my_params(self)
    self.output_len = output_len
    self.W = subcol.add_parameters(dim=(hidden_dim, output_len),
                                   init=param_init.initializer((hidden_dim, output_len)))
    self.pos_enc_max = pos_enc_max

  def __call__(self, x):
    x_T = x.as_transposed_tensor()
    scores = x_T * dy.parameter(self.W)
    if x.mask is not None:
      scores = x.mask.add_to_tensor_expr(scores, multiplicator=-100.0, time_first=True)
    if self.pos_enc_max:
      seq_len = x_T.dim()[0][0]
      pos_enc = np.zeros((seq_len, self.output_len))
      for k in range(self.output_len):
        for s in range(seq_len):
          pos_enc[s, k] = (1.0 - k / self.output_len) * (
                    1.0 - s / self.pos_enc_max) + k / self.output_len * s / self.pos_enc_max
      scores = dy.cmult(scores, dy.inputTensor(pos_enc))
    attention = dy.softmax(scores)
    output_expr = x.as_tensor() * attention
    return expression_sequence.ExpressionSequence(expr_tensor=output_expr, mask=None)
