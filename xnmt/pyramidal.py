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
               downsampling_method="concat"):
    """
    :param num_layers: depth of the PyramidalRNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    :param downsampling_method: how to perform downsampling (concat|skip)
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    self.serialize_params = [num_layers, input_dim, hidden_dim, model]
    self.builder_layers = []
    self.downsampling_method = downsampling_method
    f = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    b = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    self.builder_layers.append((f, b))
    for _ in xrange(num_layers - 1):
      layer_input_dim = hidden_dim if downsampling_method=="skip" else hidden_dim*2
      f = rnn_builder_factory(1, layer_input_dim, hidden_dim / 2, model)
      b = rnn_builder_factory(1, layer_input_dim, hidden_dim / 2, model)
      self.builder_layers.append((f, b))

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
      if self.downsampling_method=="concat" and len(es)%2!=0:
        zero_pad = dy.inputTensor(np.zeros(es[0].dim()[0]+(es[0].dim()[1],)), batched=True)
        es.append(zero_pad)
      try:
        fs = fb.initial_state().transduce(es)
      except:
        fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      if layer_i < len(self.builder_layers) - 1:
        if self.downsampling_method=="skip":
          es = [dy.concatenate([f, b]) for f, b in zip(fs[::2], bs[::2][::-1])]
        elif self.downsampling_method=="concat":
          es = [dy.concatenate([fs[i],bs[len(es)-2-i],fs[i+1],bs[len(es)-1-i]]) for i in range(0, len(es), 2)]
        else:
          raise RuntimeError("unknown downsampling_method %s" % self.downsampling_method)
      else:
        es = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return es

  def initial_state(self):
    return PseudoState(self)
