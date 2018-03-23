import numpy as np
import dynet as dy

from xnmt.batcher import Mask
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.expression_sequence import ExpressionSequence
from xnmt.linear import Linear
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.serialize.serializable import Serializable
from xnmt.transducer import SeqTransducer


class WeightNoise(object):
  def __init__(self, std):
    self.std = std

  def __call__(self, p, train=True):
    """
    Args:
      p: DyNet parameter (not expression)
      train (bool): only apply noise if True
    Returns:
      dynet.Expression: DyNet expression with weight noise applied if self.std > 0
    """
    p_expr = dy.parameter(p)
    if self.std > 0.0 and train:
      p_expr = dy.noise(p_expr, self.std)
    return p_expr


class ResidualSeqTransducer(SeqTransducer, Serializable):
  """
  Adds a residual connection.

  According to https://arxiv.org/pdf/1603.05027.pdf it is preferable to keep the shortcut
  connection pure (i.e., None), although it might be necessary to insert a linear transform to make
  layer sizes match, which can be done via the plain_resizer parameter
  (see advice here: https://github.com/fchollet/keras/issues/2608 )

  Args:
    shortcut_operation (Optional[SeqTransducer]):
    transform_operation (SeqTransducer):
  """
  yaml_tag = u'!ResidualSeqTransducer'

  def __init__(self, shortcut_operation=None, transform_operation=None):
    register_handler(self)
    self.shortcut_operation = shortcut_operation
    self.transform_operation = transform_operation

  def __call__(self, es):
    plain_es = es
    if self.shortcut_operation:
      plain_es = self.shortcut_operation(plain_es)
    transformed_es = self.transform_operation(es)
    if plain_es.dim() != transformed_es.dim():
      raise ValueError("residual connections need matching shortcut / output dimensions, got: %s and %s" % (
      plain_es.dim(), transformed_es.dim()))
    if self.shortcut_operation:
      self._final_states = [trans_f + shortc_f for trans_f, shortc_f in zip(self.transform_operation.get_final_states(),
                                                                            self.shortcut_operation.get_final_states())]
    else:
      self._final_states = self.transform_operation.get_final_states()
    return ExpressionSequence(expr_tensor=plain_es.as_tensor() + transformed_es.as_tensor(),
                              mask=plain_es.mask, tensor_transposed=plain_es.tensor_transposed)

  def get_final_states(self):
    return self._final_states


class TimePadder(object):
  """
  Pads ExpressionSequence along time axis.

  Args:
    mode (str): "zero" | "repeat_last"
  """

  def __init__(self, mode="zero"):
    self.mode = mode

  def __call__(self, es, pad_len):
    """
    Args:
      es (ExpressionSequence): input expression sequence
      pad_len (int): how much to pad
    Returns:
      ExpressionSequence: padded version of input, with padded items indicated as masked
    """
    assert not es.tensor_transposed
    #     time_dim = len(es.dim()[0])-1
    single_pad_dim = list(es.dim()[0])[:-1]
    #     single_pad_dim[time_dim] = 1
    batch_size = es.dim()[1]
    if self.mode == "zero":
      single_pad = dy.zeros(tuple(single_pad_dim), batch_size=batch_size)
    elif self.mode == "repeat_last":
      single_pad = es[-1]
    mask = es.mask
    if mask is not None:
      mask_dim = (mask.np_arr.shape[0], pad_len)
      mask = Mask(np.append(mask.np_arr, np.ones(mask_dim), axis=1))
    if es.has_list():
      es_list = es.as_list()
      es_list.extend([single_pad] * pad_len)
      return ExpressionSequence(expr_list=es_list, mask=mask)
    else:
      raise NotImplementedError("tensor padding not implemented yet")


class LayerNorm(object):
  def __init__(self, d_hid, model):
    self.p_g = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(1.0))
    self.p_b = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(0.0))

  def __call__(self, x):
    g = dy.parameter(self.p_g)
    b = dy.parameter(self.p_b)
    return dy.layer_norm(x, g, b)


class TimeDistributed(object):
  def __call__(self, x):
    batch_size = x[0].dim()[1]
    model_dim = x[0].dim()[0][0]
    seq_len = len(x)
    total_words = seq_len * batch_size
    input_tensor = x.as_tensor()
    return dy.reshape(input_tensor, (model_dim,), batch_size=total_words)


class PositionwiseFeedForward(object):
  def __init__(self, input_dim, hidden_dim, model, nonlinearity="rectify", param_init=GlorotInitializer()):
    """
    Args:
        input_dim(int): the size of input for the first-layer of the FFN.
        hidden_dim(int): the hidden layer size of the second-layer
                          of the FNN.
    """
    self.w_1 = Linear(input_dim, hidden_dim, model, param_init=param_init)
    self.w_2 = Linear(hidden_dim, input_dim, model, param_init=param_init)
    self.layer_norm = LayerNorm(input_dim, model)
    self.nonlinearity = getattr(dy, nonlinearity)

  def __call__(self, x, p):
    residual = x
    output = self.w_2(self.nonlinearity(self.w_1(x)))
    if p > 0.0:
      output = dy.dropout(output, p)
    return self.layer_norm(output + residual)


class PositionwiseLinear(object):
  def __init__(self, input_dim, hidden_dim, model, param_init=GlorotInitializer(), bias_init=ZeroInitializer()):
    """
    Args:
        input_dim(int): the size of input for the first-layer of the FFN.
        hidden_dim(int): the hidden layer size of the second-layer
                          of the FNN.
    """
    self.w_1 = Linear(input_dim, hidden_dim, model, param_init=param_init, bias_init=bias_init)

  def __call__(self, x):
    return self.w_1(x)


