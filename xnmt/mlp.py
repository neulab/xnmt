import dynet as dy

class MLP(object):
  def __init__(self, input_dim, hidden_dim, output_dim, model):
    self.W1 = model.add_parameters((hidden_dim, input_dim))
    self.b1 = model.add_parameters(hidden_dim)
    self.W2 = model.add_parameters((output_dim, hidden_dim))
    self.b2 = model.add_parameters(output_dim)

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    W2 = dy.parameter(self.W2)
    b1 = dy.parameter(self.b1)
    b2 = dy.parameter(self.b2)

    h = dy.tanh(W1 * input_expr + b1)
    return W2 * h + b2
