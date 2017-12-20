import dynet as dy
from xnmt.initializer import LeCunUniform as linear_init


class Linear(object):
  def __init__(self, input_dim, output_dim, model):
    self.W1 = model.add_parameters((output_dim, input_dim), init=linear_init(input_dim))
    self.b1 = model.add_parameters(output_dim)

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)

    return dy.affine_transform([b1, W1, input_expr])


class LinearNoBias(object):
  def __init__(self, input_dim, output_dim, model):
    self.W1 = model.add_parameters((output_dim, input_dim), init=linear_init(input_dim))
    self.output_dim = output_dim

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    b1 = dy.zeros(self.output_dim)

    return dy.affine_transform([b1, W1, input_expr])