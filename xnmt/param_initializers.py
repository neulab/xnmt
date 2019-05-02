import math
from typing import Sequence, Tuple
import numbers

import numpy as np

import xnmt
import xnmt.tensor_tools as tt
from xnmt.persistence import serializable_init, Serializable

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch.nn.init

@xnmt.require_dynet
class ParamInitializerDynet(object):
  """
  A parameter initializer that delegates to the DyNet initializers and possibly
  performs some extra configuration.
  """

  def initializer(self,
                  dim: Tuple[numbers.Integral],
                  is_lookup: bool = False,
                  num_shared: numbers.Integral = 1) -> 'dy.Initializer':
    """
    Return initializer.

    Args:
      dim: dimension of parameter tensor
      is_lookup: True if parameters are a lookup matrix
      num_shared: Indicates if one parameter object holds multiple matrices
    Returns:
      a dynet initializer object
    """
    raise NotImplementedError("subclasses must implement initializer()")
  def initializer_pos(self,
                      index: numbers.Integral,
                      dim: Tuple[numbers.Integral],
                      is_lookup: bool = False,
                      num_shared: numbers.Integral = 1) -> 'dy.Initializer':
    """
    Return initializer at given position. Default is to use same initializer across positions, unless InitializerSequence is used.
    """
    return self.initializer(dim=dim, is_lookup=is_lookup, num_shared=num_shared)

@xnmt.require_torch
class ParamInitializerTorch(object):
  """
  A parameter initializer that delegates to the DyNet initializers and possibly
  performs some extra configuration.
  """

  def initialize(self, weights: tt.Tensor) -> None:
    """
    Initialize given weights.

    Args:
      weights: parameter tensor to be initialized
    """
    raise NotImplementedError("subclasses must implement initializer()")
  def initialize_pos(self, index: numbers.Integral, weights: tt.Tensor) -> None:
    """
    Initialize using position-specific initializer. Default is to use same initializer across positions, unless InitializerSequence is used.
    """
    return self.initialize(weights=weights)

ParamInitializer = xnmt.resolve_backend(ParamInitializerDynet, ParamInitializerTorch)

class InitializerSequence(Serializable, ParamInitializer):
  """
  Sequence of position-specific initializers.

  This can be used when a componenent has several parameter tensors that should each be initialized using a different
  initializer. Examples would be components with multiple layers, and/or several sets of weight matrices that serve
  different purposes.

  The most commonly needed use case of this may be the case of a NumpyInitializer, where one wants to manually specify
  all network weights using respective numpy arrays.

  Args:
    sequence: sequence of initializers
  """
  yaml_tag = "!InitializerSequence"
  @serializable_init
  def __init__(self, sequence: Sequence[ParamInitializer]):
    self.sequence = sequence
  def initialize_pos(self, index, *args, **kwargs):
    if index >= len(self.sequence):
      raise ValueError(f"initializer sequence of {len(self.sequence)} is too short")
    return self.sequence[index].initialize(*args, **kwargs)
  def initializer_pos(self, index: numbers.Integral, *args, **kwargs):
    if index >= len(self.sequence):
      raise ValueError(f"initializer sequence of {len(self.sequence)} is too short")
    return self.sequence[index].initializer(*args, **kwargs)


@xnmt.require_dynet
class NormalInitializer(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.NormalInitializer':
    return dy.NormalInitializer(mean=self.mean, var=self.var)

@xnmt.require_dynet
class UniformInitializer(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.UniformInitializer':
    return dy.UniformInitializer(scale=self.scale)

@xnmt.require_dynet
class ConstInitializerDynet(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.ConstInitializer':
    return dy.ConstInitializer(c=self.c)

@xnmt.require_torch
class ConstInitializerTorch(ParamInitializerTorch, Serializable):
  """
  Initialize the parameters with a constant value.

  Args:
    c: Value to initialize the parameters
  """
  yaml_tag = "!ConstInitializer"

  @serializable_init
  def __init__(self, c: numbers.Real) -> None:
    self.c = c

  def initialize(self, weights: tt.Tensor) -> None:
    torch.nn.init.constant_(weights, val=self.c)

ConstInitializer = xnmt.resolve_backend(ConstInitializerDynet, ConstInitializerTorch)

class ZeroInitializer(ConstInitializer, Serializable):
  """
  Initializes parameter matrix to zero (most appropriate for bias parameters).
  """
  yaml_tag="!ZeroInitializer"

  @serializable_init
  def __init__(self) -> None:
    self.c=0.0

@xnmt.require_dynet
class GlorotInitializerDynet(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.UniformInitializer':
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

@xnmt.require_torch
class GlorotInitializerTorch(ParamInitializerTorch, Serializable):
  """
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

  def initialize(self, weights: tt.Tensor) -> None:
    torch.nn.init.xavier_uniform_(weights, gain = self.gain)

GlorotInitializer = xnmt.resolve_backend(GlorotInitializerDynet, GlorotInitializerTorch)



@xnmt.require_dynet
class FromFileInitializer(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.FromFileInitializer':
    return dy.FromFileInitializer(fname=self.fname)

@xnmt.require_dynet
class NumpyInitializerDynet(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.NumpyInitializer':
    if dim != self.array.shape:
      raise ValueError(f"expected same dims, got: {dim} != {self.array.shape}")
    return dy.NumpyInitializer(array=self.array)

@xnmt.require_torch
class NumpyInitializerTorch(ParamInitializerTorch, Serializable):
  """
  Initialize from numpy array.

  Args:
    array: Numpy array
  """
  yaml_tag = "!NumpyInitializer"

  @serializable_init
  def __init__(self, array: np.ndarray) -> None:
    self.array = array

  def initialize(self, weights: tt.Tensor) -> None:
    if weights.size() != self.array.shape:
      raise ValueError(f"Assuming equal dims, got: {weights.size()} != {self.array.shape}")
    weights.data = torch.Tensor(self.array).to(xnmt.device)

NumpyInitializer = xnmt.resolve_backend(NumpyInitializerDynet, NumpyInitializerTorch)


@xnmt.require_dynet
class LeCunUniformInitializer(ParamInitializerDynet, Serializable):
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

  def initializer(self, dim: Tuple[numbers.Integral], is_lookup: bool = False, num_shared: numbers.Integral = 1) -> 'dy.UniformInitializer':
    if is_lookup:
      fan_in = dim[0]
    else:
      fan_in = dim[-1]
    s = self.scale * np.sqrt(3. / fan_in)
    return dy.UniformInitializer(s)


