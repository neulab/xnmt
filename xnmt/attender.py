import logging
logger = logging.getLogger('xnmt')

import math
import dynet as dy

from xnmt.param_collection import ParamManager
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.serialize.serializable import Serializable, Ref, Path, bare
from xnmt.serialize.serializer import serializable_init

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
    input_dim (int): input dimension
    state_dim (int): dimension of state inputs
    hidden_dim (int): hidden MLP dimension
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  '''

  yaml_tag = '!MlpAttender'

  @serializable_init
  def __init__(self,
               input_dim=Ref(Path("exp_global.default_layer_dim")),
               state_dim=Ref(Path("exp_global.default_layer_dim")),
               hidden_dim=Ref(Path("exp_global.default_layer_dim")),
               param_init=Ref(Path("exp_global.param_init"), default=bare(GlorotInitializer)),
               bias_init=Ref(Path("exp_global.bias_init"), default=bare(ZeroInitializer))):
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = ParamManager.my_subcollection(self)
    self.pW = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.pb = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.pU = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))
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

  yaml_tag = '!DotAttender'

  @serializable_init
  def __init__(self, scale:bool=True):
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
    exp_global (ExpGlobal): ExpGlobal object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
    input_dim (int): input dimension; if None, use exp_global.default_layer_dim
    state_dim (int): dimension of state inputs; if None, use exp_global.default_layer_dim
    param_init (ParamInitializer): how to initialize weight matrices; if None, use ``exp_global.param_init``
  '''

  yaml_tag = '!BilinearAttender'

  serializable_init
  def __init__(self,
               input_dim=Ref(Path("exp_global.default_layer_dim")),
               state_dim=Ref(Path("exp_global.default_layer_dim")),
               param_init=Ref(Path("exp_global.param_init"), default=bare(GlorotInitializer))):
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = ParamManager.my_subcollection(self)
    self.pWa = param_collection.add_parameters((input_dim, state_dim), init=param_init.initializer((input_dim, state_dim)))
    self.curr_sent = None

  def init_sent(self, sent):
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

