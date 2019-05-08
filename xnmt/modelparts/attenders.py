import math
import numbers

import numpy as np

import xnmt
import xnmt.tensor_tools as tt
from xnmt import logger
from xnmt import expression_seqs, events, param_collections, param_initializers
from xnmt.persistence import serializable_init, Serializable, Ref, bare

if xnmt.backend_dynet:
  import dynet as dy

if xnmt.backend_torch:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

class Attender(object):
  """
  A template class for functions implementing attention.
  """

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
    """Args:
         sent: the encoder states, aka keys and values. Usually but not necessarily an :class:`expression_seqs.ExpressionSequence`
    """
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    """ Compute attention weights.

    Args:
      state: the current decoder state, aka query, for which to compute the weights.
    Returns:
      DyNet expression containing normalized attention scores
    """
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')

  def calc_context(self, state: tt.Tensor, attention: tt.Tensor = None) -> tt.Tensor:
    """ Compute weighted sum.

    Args:
      state: the current decoder state, aka query, for which to compute the weighted sum.
      attention: the attention vector to use. if not given it is calculated from the state.
    """
    attention = attention if attention is not None else self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    if xnmt.backend_dynet:
      return I * attention
    else:
      return torch.matmul(attention,I)

@xnmt.require_dynet
class MlpAttenderDynet(Attender, Serializable):
  """
  Implements the attention model of Bahdanau et. al (2014)

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices. In case of an InitializerSequence, the order is pW, pV, pU
    bias_init: how to initialize bias vectors
  """

  yaml_tag = '!MlpAttender'


  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    my_params = param_collections.ParamManager.my_params(self)
    self.pW = my_params.add_parameters((hidden_dim, input_dim), init=param_init[0].initializer((hidden_dim, input_dim)))
    self.pV = my_params.add_parameters((hidden_dim, state_dim), init=param_init[1].initializer((hidden_dim, state_dim)))
    self.pb = my_params.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.pU = my_params.add_parameters((1, hidden_dim), init=param_init[2].initializer((1, hidden_dim)))
    self.curr_sent = None
    self.attention_vecs = None
    self.WI = None

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
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

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    h = dy.tanh(dy.colwise_add(WI, V * state))
    scores = dy.transpose(U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

@xnmt.require_torch
class MlpAttenderTorch(Attender, Serializable):
  """
  Implements the attention model of Bahdanau et. al (2014)

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = '!MlpAttender'


  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    my_params = param_collections.ParamManager.my_params(self)
    self.linear_context = nn.Linear(input_dim, hidden_dim, bias=False).to(xnmt.device)
    self.linear_query = nn.Linear(state_dim, hidden_dim, bias=True).to(xnmt.device)
    self.pU = nn.Linear(hidden_dim, 1, bias=False).to(xnmt.device)
    my_params.append(self.linear_context)
    my_params.append(self.linear_query)
    my_params.append(self.pU)
    my_params.init_params(param_init, bias_init)

    self.curr_sent = None
    self.attention_vecs = None
    self.WI = None

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    self.WI = self.linear_context(I)

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    h = torch.tanh(WI + self.linear_query(state).unsqueeze(1))
    scores = self.pU(h).transpose(1,2)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = F.softmax(scores,dim=-1)
    self.attention_vecs.append(normalized)
    return normalized

MlpAttender = xnmt.resolve_backend(MlpAttenderDynet, MlpAttenderTorch)


@xnmt.require_dynet
class DotAttenderDynet(Attender, Serializable):
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
  """

  yaml_tag = '!DotAttender'

  @serializable_init
  def __init__(self,
               scale: bool = True) -> None:
    self.curr_sent = None
    self.scale = scale
    self.attention_vecs = []

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = dy.transpose(self.curr_sent.as_tensor())

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    scores = self.I * state
    if self.scale:
      scores /= math.sqrt(state.dim()[0][0])
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

@xnmt.require_torch
class DotAttenderTorch(Attender, Serializable):
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
  """

  yaml_tag = '!DotAttender'

  @serializable_init
  def __init__(self,
               scale: bool = True) -> None:
    self.curr_sent = None
    self.scale = scale
    self.attention_vecs = []

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor().transpose(1,2)

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    # https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/attentions.py#L85
    scores = torch.matmul(state.unsqueeze(1), self.I)
    if self.scale:
      scores /= math.sqrt(tt.hidden_size(state))
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = F.softmax(scores, dim=2)
    self.attention_vecs.append(normalized)
    return normalized

DotAttender = xnmt.resolve_backend(DotAttenderDynet, DotAttenderTorch)


@xnmt.require_dynet
class BilinearAttender(Attender, Serializable):
  """
  Implements a bilinear attention, equivalent to the 'general' linear
  attention of https://arxiv.org/abs/1508.04025

  Args:
    input_dim: input dimension; if None, use exp_global.default_layer_dim
    state_dim: dimension of state inputs; if None, use exp_global.default_layer_dim
    param_init: how to initialize weight matrices; if None, use ``exp_global.param_init``
  """

  yaml_tag = '!BilinearAttender'

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))) -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    my_params = param_collections.ParamManager.my_params(self)
    self.pWa = my_params.add_parameters((input_dim, state_dim), init=param_init.initializer((input_dim, state_dim)))
    self.curr_sent = None

  def init_sent(self, sent: expression_seqs.ExpressionSequence) -> None:
    self.curr_sent = sent
    self.attention_vecs = []
    self.I = self.curr_sent.as_tensor()

  # TODO(philip30): Please apply masking here
  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    logger.warning("BilinearAttender does currently not do masking, which may harm training results.")
    Wa = dy.parameter(self.pWa)
    scores = (dy.transpose(state) * Wa) * self.I
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return dy.transpose(normalized)

@xnmt.require_dynet
class LatticeBiasedMlpAttender(MlpAttender, Serializable):
  """
  Modified MLP attention, where lattices are assumed as input and the attention is biased toward confident nodes.

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  yaml_tag = '!LatticeBiasedMlpAttender'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    super().__init__(input_dim=input_dim, state_dim=state_dim, hidden_dim=hidden_dim, param_init=param_init,
                     bias_init=bias_init)

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_sent_bias = np.full((src.sent_len(), 1, src.batch_size()), -1e10)
    for batch_i, lattice_batch_elem in enumerate(src):
      for node_id in lattice_batch_elem.nodes:
        self.cur_sent_bias[node_id, 0, batch_i] = lattice_batch_elem.graph[node_id].marginal_log_prob
    self.cur_sent_bias_expr = None

  def calc_attention(self, state: tt.Tensor) -> tt.Tensor:
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    h = dy.tanh(dy.colwise_add(WI, V * state))
    scores = dy.transpose(U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -1e10)
    if self.cur_sent_bias_expr is None: self.cur_sent_bias_expr = dy.inputTensor(self.cur_sent_bias, batched=True)
    normalized = dy.softmax(scores + self.cur_sent_bias_expr)
    self.attention_vecs.append(normalized)
    return normalized

