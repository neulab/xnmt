import math
from typing import Tuple
import numbers

import numpy as np
import dynet as dy

from xnmt.persistence import serializable_init, Serializable

class ParamInitializer(object):
  """
  A parameter initializer that delegates to the DyNet initializers and possibly
  performs some extra configuration.
  """

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.Initializer':
    """
    Args:
      dim: dimension of parameter tensor
      is_lookup: True if parameters are a lookup matrix
      num_shared: Indicates if one parameter object holds multiple matrices
    Returns:
      a dynet initializer object
    """
    raise NotImplementedError("subclasses must implement initializer()")

#### DYNET DEFAULT INITIALIZERS ####

class NormalInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's NormalInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.NormalInitializer

  Initialize the parameters with a gaussian distribution.

  Args:
    mean: Mean of the distribution
    var: Variance of the distribution
  """
  yaml_tag = "!NormalInitializer"

  @serializable_init
  def __init__(self, mean: numbers.Real = 0, var: numbers.Real = 1) -> None:
    self.mean = mean
    self.var = var

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.NormalInitializer:
    return dy.NormalInitializer(mean=self.mean, var=self.var)

class UniformInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's UniformInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.UniformInitializer

  Initialize the parameters with a uniform distribution.
  Args:
    scale: Parameters are sampled from :math:`\\mathcal U([-\\texttt{scale},\\texttt{scale}])`
  """
  yaml_tag = "!UniformInitializer"

  @serializable_init
  def __init__(self, scale: numbers.Real) -> None:
    self.scale = scale

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.UniformInitializer:
    return dy.UniformInitializer(scale=self.scale)

class ConstInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's ConstInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.ConstInitializer

  Initialize the parameters with a constant value.

  Args:
    c: Value to initialize the parameters
  """
  yaml_tag = "!ConstInitializer"

  @serializable_init
  def __init__(self, c: numbers.Real) -> None:
    self.c = c

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.ConstInitializer:
    return dy.ConstInitializer(c=self.c)

class GlorotInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's GlorotInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.GlorotInitializer
  
  Initializes the weights according to `Glorot & Bengio (2011) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ 
    
    If the dimensions of the parameter matrix are :math:`m,n`, the weights are sampled from :math:`\\mathcal U([-g\\sqrt{\\frac{6}{m+n}},g\\sqrt{\\frac{6}{m+n}}])`
    
    The gain :math:`g` depends on the activation function : 

    * :math:`\\text{tanh}` : 1.0
    * :math:`\\text{ReLU}` : 0.5
    * :math:`\\text{sigmoid}` : 4.0
    * Any smooth function :math:`f` : :math:`\\frac{1}{f'(0)}`

  In addition to the DyNet class, this also supports the case where one parameter object stores several matrices (as is popular for computing LSTM gates, for instance).

    *Note:* This is also known as **Xavier initialization**

  Args:
    gain: Gain (Depends on the activation function)
  """
  yaml_tag = "!GlorotInitializer"

  @serializable_init
  def __init__(self, gain: numbers.Real = 1.0) -> None:
    self.gain = gain

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.UniformInitializer:
    """
    Args:
      dim: dimensions of parameter tensor
      is_lookup : Whether the parameter is a lookup parameter
      num_shared: If > 1, treat the first dimension as spanning multiple matrices, each of which is initialized individually
    Returns:
      a dynet initializer object
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

  Args:
    fname: File name
  """
  yaml_tag = "!FromFileInitializer"

  @serializable_init
  def __init__(self, fname: str) -> None:
    self.fname = fname

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.FromFileInitializer:
    return dy.FromFileInitializer(fname=self.fname)

class NumpyInitializer(ParamInitializer, Serializable):
  """
  Wraps DyNet's NumpyInitializer: http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.NumpyInitializer

  Initialize from numpy array

  Alternatively, use ``ParameterCollection.parameters_from_numpy()``

  Args:
    array: Numpy array
  """
  yaml_tag = "!NumpyInitializer"

  @serializable_init
  def __init__(self, array: np.ndarray) -> None:
    self.array = array

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.NumpyInitializer:
    return dy.NumpyInitializer(array=self.array)


#### XNMT CUSTOM INITIALIZERS ####

class ZeroInitializer(ParamInitializer, Serializable):
  """
  Initializes parameter matrix to zero (most appropriate for bias parameters).
  """
  yaml_tag="!ZeroInitializer"

  @serializable_init
  def __init__(self) -> None:
    pass

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.ConstInitializer:
    return dy.ConstInitializer(c=0.0)

class LeCunUniformInitializer(ParamInitializer, Serializable):
  """
  Reference: LeCun 98, Efficient Backprop
  http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

  Args:
    scale: scale
  """
  yaml_tag = "!LeCunUniformInitializer"

  @serializable_init
  def __init__(self, scale: numbers.Real = 1.0) -> None:
    self.scale = scale

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> dy.UniformInitializer:
    if is_lookup:
      fan_in = dim[0]
    else:
      fan_in = dim[-1]
    s = self.scale * np.sqrt(3. / fan_in)
    return dy.UniformInitializer(s)
