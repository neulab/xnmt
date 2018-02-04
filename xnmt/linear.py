import dynet as dy
from xnmt.initializer import LeCunUniform


class Linear(object):
  def __init__(self, input_dim, output_dim, model, bias=True, init=None):
    self.bias = bias
    self.output_dim = output_dim
    init_w, init_b = None, None

    if init == 'LeCunUniform':
      init_w = LeCunUniform(input_dim)
      init_b = LeCunUniform(output_dim)

    self.W1 = model.add_parameters((output_dim, input_dim), init=init_w)
    if self.bias:
      self.b1 = model.add_parameters(output_dim, init=init_b)

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    if self.bias:
      b1 = dy.parameter(self.b1)
    else:
      b1 = dy.zeros(self.output_dim)

    return dy.affine_transform([b1, W1, input_expr])
