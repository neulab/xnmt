import dynet as dy


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

    def __repr__(self):
        return "LayerNorm module"


class PositionwiseFeedForward(object):
    """ A two-layer Feed-Forward-Network."""

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

    def __call__(self, x, p=0.):
        residual = x
        output = dy.dropout(self.w_2(dy.rectify(self.w_1(x))), p)
        return self.layer_norm(output + residual)
