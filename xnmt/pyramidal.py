import dynet as dy
from residual import PseudoState
import numpy as np

class PyramidalRNNBuilder(object):
  """
  Builder for pyramidal RNNs that delegates to regular RNNs and wires them together.
  See https://arxiv.org/abs/1508.01211
  
  Every layer (except the first) reduces sequence length by factor 2.  
  
      builder = PyramidalRNNBuilder(4, 128, 100, model, VanillaLSTMBuilder)
      [o1,o2,o3] = builder.transduce([i1,i2,i3])
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory,
               downsampling_method="concat", reduce_factor=2):
    """
    :param num_layers: depth of the PyramidalRNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    :param downsampling_method: how to perform downsampling (concat|skip)
    :param reduce_factor: integer, or list of ints (different skip for each layer)
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    assert type(reduce_factor)==int or (type(reduce_factor)==list and len(reduce_factor)==num_layers-1)
    self.serialize_params = [num_layers, input_dim, hidden_dim, model]
    self.builder_layers = []
    self.downsampling_method = downsampling_method
    self.reduce_factor = reduce_factor
    f = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    b = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    self.builder_layers.append((f, b))
    for _ in range(num_layers - 1):
      layer_input_dim = hidden_dim if downsampling_method=="skip" else hidden_dim*reduce_factor
      f = rnn_builder_factory(1, layer_input_dim, hidden_dim / 2, model)
      b = rnn_builder_factory(1, layer_input_dim, hidden_dim / 2, model)
      self.builder_layers.append((f, b))

  def reduce_factor_for_layer(self, layer_i):
    if layer_i >= len(self.builder_layers)-1:
      return 1
    elif type(self.reduce_factor)==int:
      return self.reduce_factor
    else:
      return self.reduce_factor[layer_i]

  def whoami(self): return "PyramidalRNNBuilder"

  def set_dropout(self, p):
    for (fb, bb) in self.builder_layers:
      fb.set_dropout(p)
      bb.set_dropout(p)
  def disable_dropout(self):
    for (fb, bb) in self.builder_layers:
      fb.disable_dropout()
      bb.disable_dropout()

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs, 
    and concatenating.
        
    :param es: an ExpressionSequence
    """
    
    es = list(es)

    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      reduce_factor = self.reduce_factor_for_layer(layer_i)
      if self.downsampling_method=="concat" and len(es)%reduce_factor!=0:
        zero_pad = dy.inputTensor(np.zeros(es[0].dim()[0]+(es[0].dim()[1],)), batched=True)
        while len(es)%reduce_factor != 0:
          es.append(zero_pad)
      fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      if layer_i < len(self.builder_layers) - 1:
        if self.downsampling_method=="skip":
          es = [dy.concatenate([f, b]) for f, b in zip(fs[::reduce_factor], bs[::reduce_factor][::-1])]
        elif self.downsampling_method=="concat":
          es_len = len(es)
          es = []
          for i in range(0, es_len, reduce_factor):
            concat_states = []
            for j in range(reduce_factor):
              concat_states.append(fs[i+j])
              concat_states.append(bs[len(es)-reduce_factor+j-i])
            es.append(dy.concatenate(concat_states))
        else:
          raise RuntimeError("unknown downsampling_method %s" % self.downsampling_method)
      else:
        es = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return es

  def initial_state(self):
    return PseudoState(self)
