import dynet as dy

from xnmt.param_collections import ParamManager
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.persistence import serializable_init, Serializable, bare, Ref

class Transform(object):
  """
  A class of transforms that change a dynet expression into another.
  """
  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    raise NotImplementedError('transform() must be implemented in subclasses of Transform')

class Identity(Transform, Serializable):
  """
  Identity transform. For use when you think it might be a better idea to
  not perform a specific transform in a place where you would normally do one.
  """
  yaml_tag = "!Identity"

  @serializable_init
  def __init__(self):
    pass

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    return input_expr

class Linear(Transform, Serializable):
  """
  Linear projection with optional bias.
  
  Args:
    input_dim (int): input dimension
    output_dim (int): hidden dimension
    bias (bool): whether to add a bias
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  """

  yaml_tag = "!Linear"

  @serializable_init
  def __init__(self,
                input_dim: int = Ref("exp_global.default_layer_dim"),
                output_dim: int = Ref("exp_global.default_layer_dim"),
                bias=True,
                param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
                bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
    self.bias = bias
    self.input_dim = input_dim
    self.output_dim = output_dim

    model = ParamManager.my_params(self)
    self.W1 = model.add_parameters((output_dim, input_dim), init=param_init.initializer((output_dim, input_dim)))
    if self.bias:
      self.b1 = model.add_parameters((output_dim,), init=bias_init.initializer((output_dim,)))

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
      return dy.affine_transform([b1, W1, input_expr])
    else:
      return W1 * input_expr

class NonLinear(Transform, Serializable):
  """
  Linear projection with optional bias and non-linearity.
  
  Args:
    input_dim (int): input dimension
    output_dim (int): hidden dimension
    bias (bool): whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  """

  yaml_tag = "!NonLinear"

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
    self.bias = bias
    self.output_dim = output_dim
    self.input_dim = input_dim
    if activation == 'tanh':
      self.activation = dy.tanh
    elif activation == 'relu':
      self.activation = dy.rectify
    elif activation == 'sigmoid':
      self.activation = dy.sigmoid
    elif activation == 'elu':
      self.activation = dy.elu
    elif activation == 'selu':
      self.activation = dy.selu
    elif activation == 'asinh':
      self.activation = dy.asinh
    elif activation == 'identity':
      def identity(x):
        return x
      self.activation = identity
    else:
      raise ValueError('Unknown activation %s' % activation)

    model = ParamManager.my_params(self)
    self.W1 = model.add_parameters((self.output_dim, self.input_dim), init=param_init.initializer((self.output_dim, self.input_dim)))
    if self.bias:
      self.b1 = model.add_parameters((self.output_dim,), init=bias_init.initializer((self.output_dim,)))

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
      return self.activation(dy.affine_transform([b1, W1, input_expr]))
    else:
      return self.activation(W1 * input_expr)

# TODO: can we come up with a more elegant way to handle things that doesn't require this?
#       currently this is necessary because of this: https://github.com/neulab/xnmt/issues/441#issuecomment-400051066
class AuxNonLinear(NonLinear, Serializable):
  """
  NonLinear with an additional auxiliary input.
  
  Args:
    input_dim (int): input dimension
    output_dim (int): hidden dimension
    aux_input_dim (int): auxiliary input dimension.
                         The actual input dimension is aux_input_dim + input_dim. This is useful
                         for when you want to do something like input feeding.
    bias (bool): whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  """

  yaml_tag = "!AuxNonLinear"

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim"),
               aux_input_dim: int = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
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

class MLP(Transform, Serializable):
  """
  A multi-layer perceptron. Defined as one or more NonLinear transforms of equal hidden
  dimension and type, then a Linear transform to the output dimension.
  """
  yaml_tag = "!MLP"

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_dim: int = Ref("exp_global.default_layer_dim"),
               bias: bool = True,
               activation: str = 'tanh',
               hidden_layers: int = 1,
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
    self.layers = []
    if hidden_layers > 0:
      self.layers = [NonLinear(input_dim=input_dim, output_dim=hidden_dim, bias=bias, activation=activation, param_init=param_init, bias_init=bias_init)]
      self.layers += [NonLinear(input_dim=hidden_dim, output_dim=hidden_dim, bias=bias, activation=activation, param_init=param_init, bias_init=bias_init) for _ in range(1,hidden_layers)]
    self.layers += [Linear(input_dim=hidden_dim, output_dim=output_dim, bias=bias, param_init=param_init, bias_init=bias_init)]

  def transform(self, expr: dy.Expression) -> dy.Expression:
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
  def __init__(self, op="rectify"):
    self.op = getattr(dy, op, None)
    if not self.op:
      raise ValueError(f"DyNet does not have an operation '{op}'.")
  def transform(self, input_expr: dy.Expression):
    return self.op(input_expr)
