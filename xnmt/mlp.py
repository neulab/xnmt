import dynet as dy
import xnmt.linear
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.serializer import serializable_init

class MLP(Serializable):
  yaml_tag = "!MLP"
  @serializable_init
  def __init__(self, input_dim, hidden_dim, output_dim):
    self.hidden = xnmt.linear.Linear(input_dim, hidden_dim)
    self.output = xnmt.linear.Linear(hidden_dim, output_dim)

  def __call__(self, input_expr):
    return self.output(dy.tanh(self.hidden(input_expr)))
