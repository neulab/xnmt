"""
This module holds normalizers for neural networks. Currently implemented are layer norm and batch norm.
"""
from typing import Optional

import dynet as dy
import numpy as np

from xnmt import batcher

from xnmt import param_collection
from xnmt.persistence import Serializable, serializable_init


class LayerNorm(Serializable):
  yaml_tag = "!LayerNorm"

  @serializable_init
  def __init__(self, d_hid):
    subcol = param_collection.ParamManager.my_params(self)
    self.p_g = subcol.add_parameters(dim=d_hid, init=dy.ConstInitializer(1.0))
    self.p_b = subcol.add_parameters(dim=d_hid, init=dy.ConstInitializer(0.0))

  def __call__(self, x):
    g = dy.parameter(self.p_g)
    b = dy.parameter(self.p_b)
    return dy.layer_norm(x, g, b)


BN_EPS = 0.1
BN_MOMENTUM = 0.1


class BatchNorm(Serializable):
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
  def __init__(self, hidden_dim: int, num_dim: int, time_first: bool = True,
               population_running_mean: Optional[np.ndarray] = None,
               population_running_std: Optional[np.ndarray] = None) -> None:
    model = param_collection.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.num_dim = num_dim
    self.time_first = time_first
    self.gamma = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(1.0))
    self.beta = model.add_parameters(dim=self.get_normalizer_dimensionality(), init=dy.ConstInitializer(0.0))
    if population_running_mean is None:
      self.population_running_mean = np.zeros((hidden_dim,))
    else:
      self.population_running_mean = population_running_mean
    if population_running_std is None:
      self.population_running_std = np.ones((hidden_dim,))
    else:
      self.population_running_std = population_running_std

  def get_normalizer_dimensionality(self):
    if self.num_dim == 1:
      return self.hidden_dim,
    elif self.num_dim == 2:
      return (1, self.hidden_dim,) if self.time_first else (self.hidden_dim, 1)
    elif self.num_dim == 3:
      if not self.time_first: raise ValueError("num_dim==3 requires time_first==True")
      return 1, 1, self.hidden_dim,
    else:
      raise NotImplementedError("BatchNorm not implemented for num_dim > 3")

  def get_stat_dimensions(self):
    if self.time_first: return list(range(self.num_dim-1))
    else: return list(range(1, self.num_dim))

  def __call__(self, input_expr: dy.Expression, train: bool, mask: Optional[batcher.Mask]=None):
    """
    Apply batch norm.

    Args:
      input_expr: input
      train: if ``True``, compute batch statistics, if ``False``, use precomputed statistics
      mask: compute statistics only over unmasked parts of the input expression
    """
    dim_in = input_expr.dim()
    param_bn_gamma = dy.parameter(self.gamma)
    param_bn_beta = dy.parameter(self.beta)
    if train:
      num_unmasked = 0
      if mask is not None:
        input_expr = mask.set_masked_to_mean(input_expr, self.time_first)
        num_unmasked = (mask.np_arr.size - np.count_nonzero(mask.np_arr)) * mask.broadcast_factor(input_expr)
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