import numbers
from typing import Optional, Sequence

import xnmt
from xnmt import param_collections, param_initializers, tensor_tools as tt
from xnmt.persistence import serializable_init, Serializable, bare, Ref

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch
  from torch import nn


class Transform(object):
  """
  A class of transforms that change a dynet expression into another.
  """
  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    raise NotImplementedError('transform() must be implemented in subclasses of Transform')

class Identity(Transform, Serializable):
  """
  Identity transform. For use when you think it might be a better idea to
  not perform a specific transform in a place where you would normally do one.
  """
  yaml_tag = "!Identity"

  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    return input_expr

@xnmt.require_dynet
class LinearDynet(Transform, Serializable):
  """
  Linear projection with optional bias.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  yaml_tag = "!Linear"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool=True,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.bias = bias
    self.input_dim = input_dim
    self.output_dim = output_dim

    my_params = param_collections.ParamManager.my_params(self)
    self.W1 = my_params.add_parameters((output_dim, input_dim), init=param_init.initializer((output_dim, input_dim)))
    if self.bias:
      self.b1 = my_params.add_parameters((output_dim,), init=bias_init.initializer((output_dim,)))

  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
      return dy.affine_transform([b1, W1, input_expr])
    else:
      return W1 * input_expr


@xnmt.require_torch
class LinearTorch(Transform, Serializable):
  """
  Linear projection with optional bias.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  yaml_tag = "!Linear"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool=True,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.bias = bias
    self.input_dim = input_dim
    self.output_dim = output_dim

    my_params = param_collections.ParamManager.my_params(self)
    self.linear = nn.Linear(in_features=input_dim,
                            out_features=output_dim,
                            bias=bias).to(xnmt.device)
    my_params.append(self.linear)
    my_params.init_params(param_init, bias_init)


  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    return self.linear(input_expr)

Linear = xnmt.resolve_backend(LinearDynet, LinearTorch)

@xnmt.require_dynet
class NonLinearDynet(Transform, Serializable):
  """
  Linear projection with optional bias and non-linearity.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = "!NonLinear"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.bias = bias
    self.output_dim = output_dim
    self.input_dim = input_dim
    self.activation = tt.activation_by_name(activation)

    my_params = param_collections.ParamManager.my_params(self)
    self.W1 = my_params.add_parameters((self.output_dim, self.input_dim), init=param_init.initializer((self.output_dim, self.input_dim)))
    if self.bias:
      self.b1 = my_params.add_parameters((self.output_dim,), init=bias_init.initializer((self.output_dim,)))

  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
      return self.activation(dy.affine_transform([b1, W1, input_expr]))
    else:
      return self.activation(W1 * input_expr)

@xnmt.require_torch
class NonLinearTorch(Transform, Serializable):
  """
  Linear projection with optional bias and non-linearity.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = "!NonLinear"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.bias = bias
    self.output_dim = output_dim
    self.input_dim = input_dim
    self.activation = tt.activation_by_name(activation)
    self.linear = nn.Linear(in_features=self.input_dim,
                            out_features=self.output_dim,
                            bias=self.bias).to(xnmt.device)
    my_params = param_collections.ParamManager.my_params(self)
    my_params.append(self.linear)
    my_params.init_params(param_init, bias_init)

  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    return self.activation(self.linear(input_expr))

NonLinear = xnmt.resolve_backend(NonLinearDynet, NonLinearTorch)

class MLP(Transform, Serializable):
  """
  A multi-layer perceptron. Defined as one or more NonLinear transforms of equal hidden
  dimension and type, then a Linear transform to the output dimension.
  """
  yaml_tag = "!MLP"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               hidden_layers: numbers.Integral = 1,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               layers: Optional[Sequence[Transform]] = None) -> None:
    self.layers = self.add_serializable_component("layers",
                                                  layers,
                                                  lambda: MLP._create_layers(num_layers=hidden_layers,
                                                                             input_dim=input_dim,
                                                                             hidden_dim=hidden_dim,
                                                                             output_dim=output_dim,
                                                                             bias=bias,
                                                                             activation=activation,
                                                                             param_init=param_init,
                                                                             bias_init=bias_init))

  @staticmethod
  def _create_layers(num_layers: numbers.Integral, input_dim: numbers.Integral, hidden_dim: numbers.Integral,
                     output_dim: numbers.Integral, bias: bool, activation: str,
                     param_init: param_initializers.ParamInitializer, bias_init: param_initializers.ParamInitializer) \
          -> Sequence[Transform]:
    layers = []
    if num_layers > 0:
      layers = [NonLinear(input_dim=input_dim, output_dim=hidden_dim, bias=bias, activation=activation,
                          param_init=param_init, bias_init=bias_init)]
      layers += [NonLinear(input_dim=hidden_dim, output_dim=hidden_dim, bias=bias, activation=activation,
                           param_init=param_init, bias_init=bias_init) for _ in range(1, num_layers)]
    layers += [Linear(input_dim=hidden_dim if num_layers>0 else input_dim,
                      output_dim=output_dim,
                      bias=bias,
                      param_init=param_init,
                      bias_init=bias_init)]
    return layers

  def transform(self, expr: tt.Tensor) -> tt.Tensor:
    for layer in self.layers:
      expr = layer.transform(expr)
    return expr

class Cwise(Transform, Serializable):
  """
  A component-wise transformation that can be an arbitrary unary DyNet operation.

  Args:
    op: arbitrary unary DyNet node
  """
  yaml_tag = "!Cwise"
  @serializable_init
  def __init__(self, op: str = "rectify") -> None:
    self.op = tt.activation_by_name(op)

  def transform(self, input_expr: tt.Tensor) -> tt.Tensor:
    return self.op(input_expr)


# TODO: can we come up with a more elegant way to handle things that doesn't require this?
#       currently this is necessary because of this: https://github.com/neulab/xnmt/issues/441#issuecomment-400051066
class AuxNonLinear(NonLinear, Serializable):
  """
  NonLinear with an additional auxiliary input.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    aux_input_dim: auxiliary input dimension.
                   The actual input dimension is aux_input_dim + input_dim.
                   This is useful for when you want to do something like input feeding.
    bias: whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = "!AuxNonLinear"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               output_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               aux_input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    original_input_dim = input_dim
    input_dim += aux_input_dim
    super().__init__(
      input_dim=input_dim,
      output_dim=output_dim,
      bias=bias,
      activation=activation,
      param_init=param_init,
      bias_init=bias_init
    )
    self.save_processed_arg("input_dim", original_input_dim)

