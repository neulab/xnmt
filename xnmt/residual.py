from __future__ import division, generators

from dynet import *


class PseudoState:
  """
  Emulates a state object for python RNN builders. This allows them to be
  used with minimal changes in code that uses dy.LSTMBuilder.
  """
  def __init__(self, network):
    self.network = network
    self._output = None

  def add_input(self, e):
    self._output = self.network.transduce([e])[0]
    return self

  def output(self):
    return self._output

  def h(self):
    raise NotImplementedError("h() is not supported on PseudoStates")

  def s(self):
    raise NotImplementedError("s() is not supported on PseudoStates")


class ResidualRNNBuilder:
  """
  Builder for RNNs that implements additional residual connections between layers: the output of each
  intermediate hidden layer is added to its output.

  input ---> hidden layer 1 ---> hidden layer 2 -+--> ... --+--> hidden layer n
                              \_________________/  \_ ... _/
  """

  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory):
    """
    @param num_layers: depth of the RNN (> 0)
    @param input_dim: size of the inputs
    @param hidden_dim: size of the outputs (and intermediate layer representations)
    @param model
    @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
    assert num_layers > 0
    self.builder_layers = []
    self.builder_layers.append(rnn_builder_factory(1, input_dim, hidden_dim, model))
    for _ in range(num_layers - 1):
      self.builder_layers.append(rnn_builder_factory(1, hidden_dim, hidden_dim, model))

  def whoami(self):
    return "ResidualRNNBuilder"

  def set_dropout(self, p):
    for l in self.builder_layers:
      l.set_dropout(p)

  def disable_dropout(self):
    for l in self.builder_layers:
      l.disable_dropout()

  def add_inputs(self, es):
    """
    Returns the list of RNNStates obtained by adding the inputs to the RNN.

    @param es: a list of Expression

    see also transduce(xs)

    .transduce(xs) is different from .add_inputs(xs) in the following way:

        .add_inputs(xs) returns a list of RNNState objects. RNNState objects can be
         queried in various ways. In particular, they allow access to the previous
         state, as well as to the state-vectors (h() and s() )

        .transduce(xs) returns a list of Expression. These are just the output
         expressions. For many cases, this suffices.
         transduce is much more memory efficient than add_inputs.
    """
    if len(self.builder_layers) == 1:
      return self.builder_layers[0].initial_state().add_inputs(es)

    es = self.builder_layers[0].initial_state().transduce(es)

    for l in self.builder_layers[1:-1]:
      es = [out + orig for (out, orig) in zip(l.initial_state().transduce(es), es)]

    return self.builder_layers[-1].initial_state().add_inputs(es)

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one.

    @param es: a list of Expression

    see also add_inputs(xs), including for explanation of differences between
    add_inputs and this function.
    """
    es = self.builder_layers[0].initial_state().transduce(es)
    if len(self.builder_layers) == 1:
      return es

    for l in self.builder_layers[1:-1]:
      es = [out + orig for (out, orig) in zip(l.initial_state().transduce(es), es)]

    return self.builder_layers[-1].initial_state().transduce(es)

  def initial_state(self):
    return PseudoState(self)


class ResidualBiRNNBuilder:
  """
  A residual network with bidirectional first layer
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory):
    assert num_layers > 1
    self.forward_layer = rnn_builder_factory(1, input_dim, hidden_dim, model)
    self.backward_layer = rnn_builder_factory(1, input_dim, hidden_dim, model)
    self.residual_network = ResidualRNNBuilder(num_layers - 1, hidden_dim, hidden_dim, model, rnn_builder_factory)

  def set_droupout(self, p):
    self.forward_layer.set_dropout(p)
    self.backward_layer.set_droupout(p)
    self.residual_network.set_dropout(p)

  def disable_droupt(self):
    self.forward_layer.disable_droupout()
    self.backward_layer.disable_dropout()
    self.residual_network.disable_dropout()

  def add_inputs(self, es):
    forward_e = self.forward_layer.initial_state().transduce(es)
    backward_e = self.backward_layer.initial_state().transduce(reversed(es))

    return self.residual_network.add_inputs(forward_e + backward_e)

  def transduce(self, es):
    forward_e = self.forward_layer.initial_state().transduce(es)
    backward_e = self.backward_layer.initial_state().transduce(reversed(es))

    return self.residual_network.transduce(forward_e + backward_e)

  def initial_state(self):
    return PseudoState(self)

