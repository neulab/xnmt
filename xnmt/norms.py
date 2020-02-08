"""
This module holds normalizers for neural networks. Currently implemented are layer norm and batch norm.
"""
from typing import List, Optional, Tuple
import numbers

import numpy as np

import xnmt
import xnmt.tensor_tools as tt
from xnmt import batchers, events, expression_seqs, param_collections
from xnmt.modelparts import transforms
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init

if xnmt.backend_dynet:
  import dynet as dy

if xnmt.backend_torch:
  import torch, torch.nn as nn

@xnmt.require_dynet
class LayerNormDynet(Serializable, transforms.Transform):
  yaml_tag = "!LayerNorm"

  @serializable_init
  def __init__(self, hidden_dim: numbers.Integral) -> None:
    my_params = param_collections.ParamManager.my_params(self)
    self.p_g = my_params.add_parameters(dim=hidden_dim, init=dy.ConstInitializer(1.0))
    self.p_b = my_params.add_parameters(dim=hidden_dim, init=dy.ConstInitializer(0.0))

  def transform(self, x: tt.Tensor) -> tt.Tensor:
    g = dy.parameter(self.p_g)
    b = dy.parameter(self.p_b)
    return dy.layer_norm(x, g, b)


@xnmt.require_torch
class LayerNormTorch(Serializable, transforms.Transform):
  yaml_tag = "!LayerNorm"

  @serializable_init
  def __init__(self, hidden_dim: numbers.Integral) -> None:
    my_params = param_collections.ParamManager.my_params(self)
    self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim).to(xnmt.device)
    my_params.append(self.layer_norm)

  def transform(self, x: tt.Tensor) -> tt.Tensor:
    return self.layer_norm(x)

LayerNorm = xnmt.resolve_backend(LayerNormDynet, LayerNormTorch)



BN_EPS = 0.1
BN_MOMENTUM = 0.1

@xnmt.require_dynet
class BatchNormDynet(Serializable, transforms.Transform, transducers.SeqTransducer):
  """
  Implements batch normalization according to Ioffe and Szegedy, 2015.

  Supports application to matrices or higher-order tensors, in which case one dimension is interpreted as the time
  dimension and sequential batch norm is applied.

  A known issue is that the running mean / std is not reverted when reverting the parameters to the best model,
  though this is unlikely to make any difference in practice.

  Reference: https://arxiv.org/pdf/1502.03167.pdf

  Args:
    hidden_dim: hidden dimension of the layer to apply batch norm on
    num_dim: order of tensor
    time_first: whether the first dimension represents time
    population_running_mean: automatically set
    population_running_std: automatically set
  """
  yaml_tag = "!BatchNorm"

  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               hidden_dim: numbers.Integral,
               num_dim: numbers.Integral,
               time_first: bool = False,
               population_running_mean: Optional[np.ndarray] = None,
               population_running_std: Optional[np.ndarray] = None) -> None:
    my_params = param_collections.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.num_dim = num_dim
    self.time_first = time_first
    self.gamma = my_params.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(1.0))
    self.beta = my_params.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(0.0))
    if population_running_mean is None:
      self.population_running_mean = np.zeros((hidden_dim,))
    else:
      self.population_running_mean = population_running_mean
    if population_running_std is None:
      self.population_running_std = np.ones((hidden_dim,))
    else:
      self.population_running_std = population_running_std

  def get_normalizer_dimensionality(self) -> Tuple[numbers.Integral]:
    if self.num_dim == 1:
      return self.hidden_dim,
    elif self.num_dim == 2:
      return (1, self.hidden_dim,) if self.time_first else (self.hidden_dim, 1)
    elif self.num_dim == 3:
      if not self.time_first: raise ValueError("num_dim==3 requires time_first==True")
      return 1, 1, self.hidden_dim,
    else:
      raise NotImplementedError("BatchNorm not implemented for num_dim > 3")

  def get_stat_dimensions(self) -> List[numbers.Integral]:
    if self.time_first: return list(range(self.num_dim-1))
    else: return list(range(1, self.num_dim))

  def transform(self, input_expr: tt.Tensor, mask: Optional[batchers.Mask]=None) -> tt.Tensor:
    """
    Apply batch norm.

    Args:
      input_expr: input
      mask: compute statistics only over unmasked parts of the input expression
    """
    dim_in = input_expr.dim()
    param_bn_gamma = dy.parameter(self.gamma)
    param_bn_beta = dy.parameter(self.beta)
    if self.train:
      num_unmasked = 0
      if mask is not None:
        input_expr = set_masked_to_mean(mask, input_expr, self.time_first)
        num_unmasked = (mask.np_arr.size - np.count_nonzero(mask.np_arr)) * broadcast_factor(mask, input_expr)
      bn_mean = dy.moment_dim(input_expr, self.get_stat_dimensions(), 1, True, num_unmasked)
      neg_bn_mean_reshaped = -dy.reshape(-bn_mean, self.get_normalizer_dimensionality())
      self.population_running_mean += (-BN_MOMENTUM) * self.population_running_mean + BN_MOMENTUM * bn_mean.npvalue()
      bn_std = dy.std_dim(input_expr, self.get_stat_dimensions(), True, num_unmasked)
      self.population_running_std += (-BN_MOMENTUM) * self.population_running_std + BN_MOMENTUM * bn_std.npvalue()
    else:
      neg_bn_mean_reshaped = -dy.reshape(dy.inputVector(self.population_running_mean), self.get_normalizer_dimensionality())
      bn_std = dy.inputVector(self.population_running_std)
    bn_numerator = input_expr + neg_bn_mean_reshaped
    bn_xhat = dy.cdiv(bn_numerator, dy.reshape(bn_std, self.get_normalizer_dimensionality()) + BN_EPS)
    bn_y = dy.cmult(param_bn_gamma, bn_xhat) + param_bn_beta # y = gamma * xhat + beta
    dim_out = bn_y.dim()
    self.save_processed_arg("population_running_mean", self.population_running_mean)
    self.save_processed_arg("population_running_std", self.population_running_std)
    assert dim_out == dim_in
    return bn_y

  def transduce(self, es: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    output = self.transform(es.as_tensor(), es.mask)
    return expression_seqs.ExpressionSequence(expr_tensor=output, mask=es.mask)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val



@xnmt.require_torch
class BatchNormTorch(Serializable, transforms.Transform, transducers.SeqTransducer):
  """
  Implements batch normalization according to Ioffe and Szegedy, 2015.

  Supports application to matrices or higher-order tensors, in which case one dimension is interpreted as the time
  dimension and sequential batch norm is applied.

  A known issue is that the running mean / std is not reverted when reverting the parameters to the best model,
  though this is unlikely to make any difference in practice.

  Reference: https://arxiv.org/pdf/1502.03167.pdf

  Args:
    hidden_dim: hidden dimension of the layer to apply batch norm on
    num_dim: order of tensor
    population_running_mean: automatically set
    population_running_std: automatically set
  """
  yaml_tag = "!BatchNorm"

  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               hidden_dim: numbers.Integral,
               num_dim: numbers.Integral,
               population_running_mean: Optional[np.ndarray] = None,
               population_running_std: Optional[np.ndarray] = None) -> None:
    my_params = param_collections.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.num_dim = num_dim
    my_params.register_parameter('weight', nn.parameter.Parameter(torch.Tensor(hidden_dim)))
    self.gamma = my_params._parameters['weight']
    my_params.register_parameter('bias', nn.parameter.Parameter(torch.Tensor(hidden_dim)))
    self.beta = my_params._parameters['bias']
    nn.init.uniform_(my_params.weight)
    nn.init.zeros_(my_params.bias)
    if population_running_mean is None:
      self.population_running_mean = np.zeros((hidden_dim,))
    else:
      self.population_running_mean = population_running_mean
    if population_running_std is None:
      self.population_running_std = np.ones((hidden_dim,))
    else:
      self.population_running_std = population_running_std

  def transform(self, input_expr: tt.Tensor, mask: Optional[batchers.Mask]=None) -> tt.Tensor:
    """
    Apply batch norm.

    Args:
      input_expr: input
      mask: compute statistics only over unmasked parts of the input expression
    """
    size_in = input_expr.size()
    if self.train:
      if mask is None:
        unmasked_input = input_expr
      else:
        mask_expr = torch.ByteTensor(1.0-mask.np_arr).to(xnmt.device)
        while mask_expr.dim() < input_expr.dim():
          mask_expr = mask_expr.unsqueeze(-1)
        unmasked_input = torch.masked_select(input_expr, mask_expr)
      flat_unmasked_input = unmasked_input.view((-1,self.hidden_dim))
      bn_mean = torch.mean(flat_unmasked_input, dim=0)
      neg_bn_mean_reshaped = -bn_mean
      self.population_running_mean += (-BN_MOMENTUM) * self.population_running_mean + BN_MOMENTUM * tt.npvalue(bn_mean)
      bn_std = torch.std(flat_unmasked_input, dim=0)
      self.population_running_std += (-BN_MOMENTUM) * self.population_running_std + BN_MOMENTUM * tt.npvalue(bn_std)
    else:
      neg_bn_mean_reshaped = -torch.Tensor(self.population_running_mean).to(xnmt.device)
      bn_std = torch.Tensor(self.population_running_std).to(xnmt.device)
    bn_numerator = input_expr + neg_bn_mean_reshaped
    bn_xhat = torch.div(bn_numerator, bn_std + BN_EPS)
    bn_y = torch.mul(self.gamma, bn_xhat) + self.beta # y = gamma * xhat + beta
    size_out = bn_y.size()
    self.save_processed_arg("population_running_mean", self.population_running_mean)
    self.save_processed_arg("population_running_std", self.population_running_std)
    assert size_out == size_in
    return bn_y

  def transduce(self, es: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    output = self.transform(es.as_tensor(), es.mask)
    return expression_seqs.ExpressionSequence(expr_tensor=output, mask=es.mask)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

BatchNorm = xnmt.resolve_backend(BatchNormDynet, BatchNormTorch)

# Batch norm helpers:

def broadcast_factor(mask: batchers.Mask, tensor_expr: tt.Tensor) -> numbers.Integral:
  """
  returns product(tensor_expr dims) / product(mask dims)
  """
  tensor_expr_size = tensor_expr.dim()[1]
  for d in tensor_expr.dim()[0]: tensor_expr_size *= d
  return tensor_expr_size / mask.np_arr.size

def mask_reshape_size(mask: batchers.Mask, tensor_dim: tuple, time_first: bool = False) -> tuple:
  if time_first:
    return list(reversed(mask.np_arr.shape[1:])) + [1] * (len(tensor_dim[0]) - len(mask.np_arr.shape) + 1) + [mask.np_arr.shape[0]]
  else:
    return [1] * (len(tensor_dim[0]) - len(mask.np_arr.shape) + 1) + list(reversed(mask.np_arr.shape))

def set_masked_to_mean(mask: batchers.Mask, tensor_expr: tt.Tensor, time_first: bool = False) -> tt.Tensor:
  """
  Set masked parts of the tensor expr to the mean of the unmasked parts.
  """
  if np.count_nonzero(mask.np_arr) == 0:
    return tensor_expr
  else:
    dim_before = tensor_expr.dim()
    reshape_size = mask_reshape_size(mask, tensor_expr.dim(), time_first)
    inv_mask_expr = dy.inputTensor(1.0 - np.reshape(mask.np_arr.transpose(), reshape_size), batched=True)
    unmasked = dy.cmult(tensor_expr, inv_mask_expr)
    unmasked_mean = unmasked
    while sum(unmasked_mean.dim()[0]) > 1: # loop because mean_dim only supports reducing up to 2 dimensions at a time
      unmasked_mean = dy.mean_dim(unmasked_mean, list(range(min(2,len(unmasked_mean.dim()[0])))), unmasked_mean.dim()[1]>1, n=1) # this is mean without normalization == sum
    unmasked_mean = dy.cdiv(unmasked_mean, dy.inputTensor(np.asarray([(mask.np_arr.size - np.count_nonzero(mask.np_arr)) * broadcast_factor(mask, tensor_expr)]), batched=False))
    mask_expr = dy.cmult(dy.inputTensor(np.reshape(mask.np_arr.transpose(), reshape_size), batched=True), unmasked_mean)
    ret = unmasked + mask_expr
    assert ret.dim() == dim_before
    return ret