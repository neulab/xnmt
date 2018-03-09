import logging
logger = logging.getLogger('xnmt')

import math
import dynet as dy

from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path

class Attender(object):
  '''
  A template class for functions implementing attention.
  '''

  def init_sent(self, sent):
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
  '''
  Implements the attention model of Bahdanau et. al (2014)
  
  Args:
    exp_global: :class:`xnmt.exp_global.ExpGlobal` object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
    input_dim (int): input dimension; if None, use exp_global.default_layer_dim
    state_dim (int): dimension of state inputs; if None, use exp_global.default_layer_dim
    hidden_dim (int): hidden MLP dimension; if None, use exp_global.default_layer_dim
  '''

  yaml_tag = u'!MlpAttender'

  def __init__(self, exp_global=Ref(Path("exp_global")), input_dim=None, state_dim=None, hidden_dim=None):
    input_dim = input_dim or exp_global.default_layer_dim
    state_dim = state_dim or exp_global.default_layer_dim
    hidden_dim = hidden_dim or exp_global.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = exp_global.dynet_param_collection.param_col
    self.pW = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb = param_collection.add_parameters(hidden_dim)
    self.pU = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent = None

  def init_sent(self, sent):
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

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention

class DotAttender(Attender, Serializable):
  '''
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762
  
  Args:
    scale (bool): whether to perform scaling
  '''

  yaml_tag = u'!DotAttender'

  def __init__(self, scale=True):
    self.curr_sent = None
    self.scale = scale
    self.attention_vecs = []

  def init_sent(self, sent):
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
  '''
  Implements a bilinear attention, equivalent to the 'general' linear
  attention of https://arxiv.org/abs/1508.04025

  Args:
    exp_global: :class:`xnmt.exp_global.ExpGlobal` object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
    input_dim (int): input dimension; if None, use exp_global.default_layer_dim
    state_dim (int): dimension of state inputs; if None, use exp_global.default_layer_dim
  '''

  yaml_tag = u'!BilinearAttender'

  def __init__(self, exp_global=Ref(Path("exp_global")), input_dim=None, state_dim=None):
    input_dim = input_dim or exp_global.default_layer_dim
    state_dim = state_dim or exp_global.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = exp_global.dynet_param_collection.param_col
    self.pWa = param_collection.add_parameters((input_dim, state_dim))
    self.curr_sent = None

  def init_sent(self, sent):
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor()

  # TODO(philip30): Please apply masking here
  def calc_attention(self, state):
    logger.warn("BilinearAttender does currently not do masking, which may harm training results.")
    Wa = dy.parameter(self.pWa)
    scores = (dy.transpose(state) * Wa) * self.I
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return dy.transpose(normalized)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    return self.I * attention

