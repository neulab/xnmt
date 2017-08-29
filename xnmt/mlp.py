import dynet as dy
import linear

class MLP(object):
  def __init__(self, input_dim, hidden_dim, output_dim, model):
    self.hidden = linear.Linear(input_dim, hidden_dim, model)
    self.output = linear.Linear(hidden_dim, output_dim, model)

  def __call__(self, input_expr):
    return self.output(dy.tanh(self.hidden(input_expr)))
