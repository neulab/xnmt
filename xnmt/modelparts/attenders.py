import math

import dynet as dy
from typing import Optional, List

from xnmt import logger
from xnmt import batchers, param_collections, param_initializers
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.expression_seqs import ExpressionSequence
from xnmt.transducers import base as transducers_base
from xnmt.param_initializers import GlorotInitializer, ParamInitializer

class Attender(object):
  """
  A template class for functions implementing attention.
  """

  def init_sent(self, sent: ExpressionSequence):
    """Args:
         sent: the encoder states, aka keys and values. Usually but not necessarily an :class:`xnmt.expression_sequence.ExpressionSequence`
    """
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, state):
    """ Compute attention weights.

    Args:
      state (dy.Expression): the current decoder state, aka query, for which to compute the weights.
    """
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')

  def calc_context(self, state):
    """ Compute weighted sum.

    Args:
      state (dy.Expression): the current decoder state, aka query, for which to compute the weighted sum.
    """
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention

  def get_last_attention(self):
    return self.attention_vecs[-1]

class MlpAttender(Attender, Serializable):
  """
  Implements the attention model of Bahdanau et. al (2014)

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """

  yaml_tag = '!MlpAttender'


  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               state_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    self.truncate_dec_batches = truncate_dec_batches
    param_collection = param_collections.ParamManager.my_params(self)
    self.pW = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.pb = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.pU = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))
    self.curr_sent = None

  def init_sent(self, sent: ExpressionSequence):
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    self.WI = dy.affine_transform([b, W, I])
    wi_dim = self.WI.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI = dy.reshape(self.WI, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    if self.truncate_dec_batches:
      if curr_sent_mask: state, WI, curr_sent_mask = batchers.truncate_batches(state, WI, curr_sent_mask)
      else: state, WI = batchers.truncate_batches(state, WI)
    h = dy.tanh(dy.colwise_add(WI, V * state))
    scores = dy.transpose(U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    if self.truncate_dec_batches: I, attention = batchers.truncate_batches(I, attention)
    return I * attention

class DotAttender(Attender, Serializable):
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
    truncate_dec_batches: currently unsupported
  """

  yaml_tag = '!DotAttender'

  @serializable_init
  def __init__(self, scale: bool = True,
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    if truncate_dec_batches: raise NotImplementedError("truncate_dec_batches not yet implemented for DotAttender")
    self.curr_sent = None
    self.scale = scale
    self.attention_vecs = []

  def init_sent(self, sent: ExpressionSequence):
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = dy.transpose(self.curr_sent.as_tensor())

  def calc_attention(self, state):
    scores = self.I * state
    if self.scale:
      scores /= math.sqrt(state.dim()[0][0])
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention

class BilinearAttender(Attender, Serializable):
  """
  Implements a bilinear attention, equivalent to the 'general' linear
  attention of https://arxiv.org/abs/1508.04025

  Args:
    input_dim: input dimension; if None, use exp_global.default_layer_dim
    state_dim: dimension of state inputs; if None, use exp_global.default_layer_dim
    param_init: how to initialize weight matrices; if None, use ``exp_global.param_init``
    truncate_dec_batches: currently unsupported
  """

  yaml_tag = '!BilinearAttender'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               state_dim: int = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    if truncate_dec_batches: raise NotImplementedError("truncate_dec_batches not yet implemented for BilinearAttender")
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = param_collections.ParamManager.my_params(self)
    self.pWa = param_collection.add_parameters((input_dim, state_dim), init=param_init.initializer((input_dim, state_dim)))
    self.curr_sent = None

  def init_sent(self, sent: ExpressionSequence):
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor()

  # TODO(philip30): Please apply masking here
  def calc_attention(self, state):
    logger.warning("BilinearAttender does currently not do masking, which may harm training results.")
    Wa = dy.parameter(self.pWa)
    scores = (dy.transpose(state) * Wa) * self.I
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return dy.transpose(normalized)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    return self.I * attention


class FixedSizeAttSeqTransducer(transducers_base.SeqTransducer, Serializable):
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
               param_init: ParamInitializer = Ref("exp_global.param_init",
                                                  default=bare(GlorotInitializer))) \
          -> None:
    subcol = ParamManager.my_params(self)
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

  def get_final_states(self) -> List[transducers_base.FinalTransducerState]:
    raise NotImplementedError('FixedSizeAttSeqTransducer.get_final_states() not implemented')

  def transduce(self, x: ExpressionSequence) -> ExpressionSequence:
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
    return ExpressionSequence(expr_tensor=output_expr, mask=None)

