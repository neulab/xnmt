import math

import numpy as np
import dynet as dy

from xnmt.serialize.serializable import Serializable

class ParamInitializer(object):
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim (tuple of int): dimension of parameter tensor
    :param is_lookup (bool): True if parameters are a lookup matrix
    :param num_shared (int): Indicates if one parameter object holds multiple matrices
    :returns: a dynet.Initializer object
    """
    raise NotImplementedError("ParamInitializer subclasses must implement initializer()")

#### DYNET DEFAULT INITIALIZERS ####

class NormalInitializer(ParamInitializer, Serializable):
  yaml_tag = "!NormalInitializer"
  def __init__(self, mean=0, var=1):
    self.mean = mean
    self.var = var
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.NormalInitializer(mean=self.mean, var=self.var)
class UniformInitializer(ParamInitializer, Serializable):
  yaml_tag = "!UniformInitializer"
  def __init__(self, scale):
    self.scale = scale
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.UniformInitializer(scale=self.scale)
class ConstInitializer(ParamInitializer, Serializable):
  yaml_tag = "!ConstInitializer"
  def __init__(self, c):
    self.c = c
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.ConstInitializer(c=self.c)
class GlorotInitializer(ParamInitializer, Serializable):
  yaml_tag = "!GlorotInitializer"
  def __init__(self, gain=1.0):
    self.gain = gain
  def initializer(self, dim, is_lookup=False, num_shared=1):
    gain = getattr(self, "gain", 1.0)
    if num_shared==1:
      return dy.GlorotInitializer(gain=gain, is_lookup=is_lookup)
    else:
      per_param_dims = list(dim)
      assert per_param_dims[0] % num_shared == 0
      per_param_dims[0] //= num_shared
      if is_lookup: per_param_dims = per_param_dims[:-1]
      scale = gain * math.sqrt(3.0 * len(per_param_dims)) / math.sqrt(sum(per_param_dims))
      return dy.UniformInitializer(scale=scale)
class FromFileInitializer(ParamInitializer, Serializable):
  yaml_tag = "!FromFileInitializer"
  def __init__(self, fname):
    self.fname = fname
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.FromFileInitializer(fname=self.fname)
class NumpyInitializer(ParamInitializer, Serializable):
  yaml_tag = "!NumpyInitializer"
  def __init__(self, array):
    self.array = array
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.NumpyInitializer(array=self.array)


#### XNMT CUSTOM INITIALIZERS ####

class ZeroInitializer(ParamInitializer, Serializable):
  yaml_tag="!ZeroInitializer"
  def initializer(self, dim, is_lookup=False, num_shared=1):
    return dy.ConstInitializer(c=0.0)

class LeCunUniformInitializer(ParamInitializer, Serializable):
  yaml_tag = "!LeCunUniformInitializer"
  """
  Reference: LeCun 98, Efficient Backprop
  http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  def __init__(self, scale=1.0):
    self.scale = scale
  def initializer(self, dim, is_lookup=False, num_shared=1):
    if is_lookup:
      fan_in = dim[0]
    else:
      fan_in = dim[-1]
    s = self.scale * np.sqrt(3. / fan_in)
    return dy.UniformInitializer(s)
