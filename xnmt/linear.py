import dynet as dy
from xnmt.param_init import GlorotInitializer, ZeroInitializer


class Linear(object):
  def __init__(self, input_dim, output_dim, model, bias=True, param_init=GlorotInitializer(), bias_init=ZeroInitializer()):
    self.bias = bias
    self.output_dim = output_dim

    self.W1 = model.add_parameters((output_dim, input_dim), init=param_init.initializer((output_dim, input_dim)))
    if self.bias:
      self.b1 = model.add_parameters((output_dim,), init=bias_init.initializer((output_dim,)))

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
    else:
      b1 = dy.zeros(self.output_dim)

    return dy.affine_transform([b1, W1, input_expr])
