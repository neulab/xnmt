"""
This module holds normalizers for neural networks. Currently implemented is layer norm, later batch norm etc. may be added.
"""

import dynet as dy

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
