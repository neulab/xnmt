import dynet as dy
import numpy as np


class Linear(object):
  def __init__(self, input_dim, output_dim, model):
    self.W1 = model.add_parameters((output_dim, input_dim))
    self.b1 = model.add_parameters(output_dim)

  def __call__(self, input_expr):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)

    return dy.affine_transform([b1, W1, input_expr])


class LayerNorm(object):
  def __init__(self, d_hid, model):
    self.p_g = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(1.0))
    self.p_b = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(0.0))

  def __call__(self, input):
    g = dy.parameter(self.p_g)
    b = dy.parameter(self.p_b)
    return dy.layer_norm(input, g, b)


class TimeDistributed(object):
  def __call__(self, input):
    batch_size = input[0].dim()[1]
    model_dim = input[0].dim()[0][0]
    seq_len = len(input)
    total_words = seq_len * batch_size
    input_tensor = input.as_tensor()
    return dy.reshape(input_tensor, (model_dim,), batch_size=total_words)


class PositionwiseFeedForward(object):
  def __init__(self, size, hidden_size, model):
    """
    Args:
        size(int): the size of input for the first-layer of the FFN.
        hidden_size(int): the hidden layer size of the second-layer
                          of the FNN.
        droput(float): dropout probability(0-1.0).
    """
    self.w_1 = Linear(size, hidden_size, model)
    self.w_2 = Linear(hidden_size, size, model)
    self.layer_norm = LayerNorm(size, model)

  def __call__(self, x, p):
    residual = x
    output = dy.dropout(self.w_2(dy.rectify(self.w_1(x))), p)
    return self.layer_norm(output + residual)


class AIAYNAdamTrainer(object):
  def __init__(self, param_col, learning_rate, dim, warmup_steps, beta_1=0.9, beta_2=0.999, eps=1e-8):
    self.optimizer = dy.AdamTrainer(param_col, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2, eps=eps)
    self.dim = dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def update(self):
    self.steps += 1
    decay = (self.dim**(-0.5)) * np.min([self.steps**(-0.5), self.steps * (self.warmup_steps**(-1.5))])
    self.optimizer.learning_rate = self.optimizer.learning_rate * decay
    self.optimizer.update()
