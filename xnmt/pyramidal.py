from dynet import *
from residual import PseudoState

class PyramidalRNNBuilder(object):
  """
  Builder for pyramidal RNNs that delegates to regular RNNs and wires them together.
  See https://arxiv.org/abs/1508.01211
  
  Every layer (except the first) reduces sequence length by factor 2.  
  
      builder = PyramidalRNNBuilder(4, 128, 100, model, VanillaLSTMBuilder)
      [o1,o2,o3] = builder.transduce([i1,i2,i3])
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory):
    """
    @param num_layers: depth of the PyramidalRNN
    @param input_dim: size of the inputs
    @param hidden_dim: size of the outputs (and intermediate layer representations)
    @param model
    @param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    self.builder_layers = []
    f = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    b = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    self.builder_layers.append((f, b))
    for _ in xrange(num_layers - 1):
      f = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
      b = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
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

  def add_inputs(self, es):
    """
    returns the list of state pairs (stateF, stateB) obtained by adding 
    inputs to both forward (stateF) and backward (stateB) RNNs.  

    @param es: a list of Expression

    see also transduce(xs)

    .transduce(xs) is different from .add_inputs(xs) in the following way:

        .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
         queried in various ways. In particular, they allow access to the previous
         state, as well as to the state-vectors (h() and s() )

        .transduce(xs) returns a list of Expression. These are just the output
         expressions. For many cases, this suffices. 
         transduce is much more memory efficient than add_inputs. 
    """
    for (fb, bb) in self.builder_layers[:-1]:
      fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      es = [concatenate([f, b]) for f, b in zip(fs[::2], bs[::2][::-1])]
    (fb, bb) = self.builder_layers[-1]
    fs = fb.initial_state().add_inputs(es)
    bs = bb.initial_state().add_inputs(reversed(es))
    return [(f, b) for f, b in zip(fs, reversed(bs))]

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs, 
    and concatenating.
        
    @param es: a list of Expression

    see also add_inputs(xs)

    .transduce(xs) is different from .add_inputs(xs) in the following way:

       .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
        queried in various ways. In particular, they allow access to the previous
        state, as well as to the state-vectors (h() and s() )

       .transduce(xs) returns a list of Expression. These are just the output
        expressions. For many cases, this suffices. 
        transduce is much more memory efficient than add_inputs. 
    """
    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      if layer_i < len(self.builder_layers) - 1:
        es = [concatenate([f, b]) for f, b in zip(fs[::2], bs[::2][:-1])]
      else:
        es = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return es

  def initial_state(self):
    return PseudoState(self)
