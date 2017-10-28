from __future__ import division, generators

import dynet as dy

from xnmt.lstm import CustomCompactLSTMBuilder, PseudoState
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState


class ResidualLSTMSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!ResidualLSTMSeqTransducer'

  def __init__(self, yaml_context, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None, bidirectional=True):
    self._final_states = None
    register_handler(self)
    model = yaml_context.dynet_param_collection.param_col
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    dropout = dropout or yaml_context.dropout
    self.dropout = dropout
    if bidirectional:
      self.builder = ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, residual_to_output)
    else:
      self.builder = ResidualRNNBuilder(layers, input_dim, hidden_dim, model, residual_to_output)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
  
  @handle_xnmt_event
  def on_start_sent(self, *args, **kwargs):
    self._final_states = None

  def __call__(self, sent):
    output = self.builder.transduce(sent)
    if not isinstance(output, ExpressionSequence):
      output = ExpressionSequence(expr_list=output)
    self._final_states = self.builder.get_final_states()
    return output

  def get_final_states(self):
    assert self._final_states is not None, "ResidualLSTMSeqTransducer.__call__() must be invoked before ResidualLSTMSeqTransducer.get_final_states()"
    return self._final_states

class ResidualRNNBuilder(object):
  """
  
  Builder for RNNs that implements additional residual connections between layers: the output of each
  intermediate hidden layer is added to its output.

  input ---> hidden layer 1 ---> hidden layer 2 -+--> ... -+---> hidden layer n ---+--->
                              \_________________/  \_ ... _/ \_(if add_to_output)_/
  """

  def __init__(self, num_layers, input_dim, hidden_dim, model, add_to_output=False):
    """
    :param num_layers: depth of the RNN (> 0)
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model:
    :param add_to_output: whether to add a residual connection to the output layer
    """
    assert num_layers > 0
    self.builder_layers = []
    self.builder_layers.append(CustomCompactLSTMBuilder(1, input_dim, hidden_dim, model))
    for _ in range(num_layers - 1):
      self.builder_layers.append(CustomCompactLSTMBuilder(1, hidden_dim, hidden_dim, model))

    self.add_to_output = add_to_output

  def whoami(self):
    return "ResidualRNNBuilder"

  def get_final_states(self):
    return self._final_states

  def set_dropout(self, p):
    for l in self.builder_layers:
      l.set_dropout(p)

  def disable_dropout(self):
    for l in self.builder_layers:
      l.disable_dropout()

  @staticmethod
  def _sum_lists(a, b):
    """
    Sums lists element-wise.
    """
    return [ea + eb for (ea, eb) in zip(a, b)]

  def add_inputs(self, es):
    """
    Returns the list of RNNStates obtained by adding the inputs to the RNN.

    :param es: a list of Expression

    see also transduce(xs)

    .transduce(xs) is different from .add_inputs(xs) in the following way:

        .add_inputs(xs) returns a list of RNNState objects. RNNState objects can be
         queried in various ways. In particular, they allow access to the previous
         state, as well as to the state-vectors (h() and s() )
         add_inputs is used for compatibility only, and the returned state will only
         support the output() operation, not h() or s().

        .transduce(xs) returns a list of Expression. These are just the output
         expressions. For many cases, this suffices.
         transduce is much more memory efficient than add_inputs.
    """
    return PseudoState(self, self.transduce(es))

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one.

    :param es: a list of Expression

    see also add_inputs(xs), including for explanation of differences between
    add_inputs and this function.
    """
    es = self.builder_layers[0].initial_state().transduce(es)
    self._final_states = [self.builder_layers[0].get_final_states()[0]]

    if len(self.builder_layers) == 1:
      return es

    for l in self.builder_layers[1:]:
      es = ExpressionSequence(expr_list=self._sum_lists(l.initial_state().transduce(es), es))
      self._final_states.append(FinalTransducerState(es[-1], l.get_final_states()[0].cell_expr()))

    last_output = self.builder_layers[-1].initial_state().transduce(es)

    if self.add_to_output:
      self._final_states.append(FinalTransducerState(last_output[-1], self.builder_layers[-1].get_final_states()[0].cell_expr()))
      return ExpressionSequence(expr_list=self._sum_lists(last_output, es))
    else:
      self._final_states.append(self.builder_layers[-1].get_final_states()[0])
      return last_output

  def initial_state(self):
    return PseudoState(self)


class ResidualBiRNNBuilder:
  """
  A residual network with bidirectional first layer
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, add_to_output=False):
    assert num_layers > 1
    assert hidden_dim % 2 == 0
    self.forward_layer = CustomCompactLSTMBuilder(1, input_dim, hidden_dim/2, model)
    self.backward_layer = CustomCompactLSTMBuilder(1, input_dim, hidden_dim/2, model)
    self.residual_network = ResidualRNNBuilder(num_layers - 1, hidden_dim, hidden_dim, model,
                                               add_to_output)

  def get_final_states(self):
    return self._final_states

  def set_dropout(self, p):
    self.forward_layer.set_dropout(p)
    self.backward_layer.set_dropout(p)
    self.residual_network.set_dropout(p)

  def disable_dropout(self):
    self.forward_layer.disable_dropout()
    self.backward_layer.disable_dropout()
    self.residual_network.disable_dropout()

  def add_inputs(self, es):
    return PseudoState(self, self.transduce(es))

  def transduce(self, es):
    forward_e = self.forward_layer.initial_state().transduce(es)
    backward_e = self.backward_layer.initial_state().transduce(ReversedExpressionSequence(es))
    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layer.get_final_states()[0].main_expr(),
                                                            self.backward_layer.get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layer.get_final_states()[0].cell_expr(),
                                                            self.backward_layer.get_final_states()[0].cell_expr()]))]

    output = self.residual_network.transduce(ExpressionSequence(expr_list=[dy.concatenate([f,b]) for f,b in zip(forward_e, ReversedExpressionSequence(backward_e))]))
    self._final_states += self.residual_network.get_final_states()
    return output

  def initial_state(self):
    return PseudoState(self)
