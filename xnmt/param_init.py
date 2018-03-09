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
  """
  Wraps DyNet's NormalInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.NormalInitializer
  
  Initialize the parameters with a gaussian distribution.
  """
  yaml_tag = "!NormalInitializer"
  def __init__(self, mean=0, var=1):
    """
    :param mean (float): Mean of the distribution
    :param var (float): Variance of the distribution
    """
    self.mean = mean
    self.var = var
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.NormalInitializer(mean=self.mean, var=self.var)
  
class UniformInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's UniformInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.UniformInitializer
  
  Initialize the parameters with a uniform distribution.
  """
  yaml_tag = "!UniformInitializer"
  def __init__(self, scale):
    """
    :param scale (float): Parameters are sampled from :math:`\mathcal U([-\\texttt{scale},\\texttt{scale}])`
    """
    self.scale = scale
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.UniformInitializer(scale=self.scale)
  
class ConstInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's ConstInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.ConstInitializer
  
  Initialize the parameters with a constant value.
  """
  yaml_tag = "!ConstInitializer"
  def __init__(self, c):
    """
    :param c (float): Value to initialize the parameters
    """
    self.c = c
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.ConstInitializer(c=self.c)
  
class GlorotInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's GlorotInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.GlorotInitializer
  
  Initializes the weights according to `Glorot & Bengio (2011) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ 
    
    If the dimensions of the parameter matrix are :math:`m,n`, the weights are sampled from :math:`\mathcal U([-g\sqrt{\\frac{6}{m+n}},g\sqrt{\\frac{6}{m+n}}])`
    
    The gain :math:`g` depends on the activation function : 

    * :math:`\\text{tanh}` : 1.0
    * :math:`\\text{ReLU}` : 0.5
    * :math:`\\text{sigmoid}` : 4.0
    * Any smooth function :math:`f` : :math:`\\frac{1}{f'(0)}`
    
  In addition to the DyNet class, this also supports the case where one parameter object stores several matrices (as is popular for computing LSTM gates, for instance).
    
    *Note:* This is also known as **Xavier initialization**  
  """
  yaml_tag = "!GlorotInitializer"
  def __init__(self, gain=1.0):
    """
    :param gain: Gain (Depends on the activation function)
    """
    self.gain = gain
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim (tuple): dimensions of parameter tensor
    :param is_lookup (bool): Whether the parameter is a lookup parameter
    :param num_shared (int): If > 1, treat the first dimension as spanning multiple matrices, each of which is initialized individually
    """
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
  """
  Wraps DyNet's FromFileInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.FromFileInitializer
  
  Initialize parameter from file.
  """
  yaml_tag = "!FromFileInitializer"
  def __init__(self, fname):
    """
    :param fname (string): File name
    """
    self.fname = fname
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.FromFileInitializer(fname=self.fname)
  
class NumpyInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's NumpyInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.NumpyInitializer
  
  Initialize from numpy array
  
  Alternatively, use ``ParameterCollection.parameters_from_numpy()``
  """
  yaml_tag = "!NumpyInitializer"
  def __init__(self, array):
    """
    :param array (np.ndarray): Numpy array
    """
    self.array = array
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.NumpyInitializer(array=self.array)


#### XNMT CUSTOM INITIALIZERS ####

class ZeroInitializer(ParamInitializer, Serializable):
  """
  Initializes parameter matrix to zero (most appropriate for bias parameters).
  """
  yaml_tag="!ZeroInitializer"
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim: (ignored)
    :param is_lookup: (ignored)
    :param num_shared: (ignored)
    """
    return dy.ConstInitializer(c=0.0)

class LeCunUniformInitializer(ParamInitializer, Serializable):
  yaml_tag = "!LeCunUniformInitializer"
  """
  Reference: LeCun 98, Efficient Backprop
  http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  def __init__(self, scale=1.0):
    """
    :param scale (float): scale
    """
    self.scale = scale
  def initializer(self, dim, is_lookup=False, num_shared=1):
    """
    :param dim (tuple): dimensions of parameter tensor
    :param is_lookup: Whether the parameter is a lookup parameter
    :param num_shared: (ignored)
    """
    if is_lookup:
      fan_in = dim[0]
    else:
      fan_in = dim[-1]
    s = self.scale * np.sqrt(3. / fan_in)
    return dy.UniformInitializer(s)
