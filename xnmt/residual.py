import dynet as dy

from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState

class PseudoState(object):
  """
  Emulates a state object for python RNN builders. This allows them to be
  used with minimal changes in code that uses dy.VanillaLSTMBuilder.
  """
  def __init__(self, network, output=None):
    self.network = network
    self._output = output

  def add_input(self, e):
    self._output = self.network.transduce([e])[0]
    return self

  def output(self):
    return self._output

  def h(self):
    raise NotImplementedError("h() is not supported on PseudoStates")

  def s(self):
    raise NotImplementedError("s() is not supported on PseudoStates")


class ResidualLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  Residual LSTM sequence transducer. Wrapper class that delegates to
  :class:`xnmt.residual.ResidualRNNBuilder` or :class:`xnmt.residual.ResidualBiRNNBuilder`,
  depending on the value of the bidirectional argument.

  Args:
    input_dim (int): size of the inputs
    layers (int): depth of the RNN (> 0)
    hidden_dim (int): size of the outputs (and intermediate layer representations)
    residual_to_output (bool): whether to add a residual connection to the output layer
    dropout (float): dropout probability
    bidirectional (bool): whether the LSTM layers should be bidirectional
    builder (Union[ResidualRNNBuilder,ResidualBiRNNBuilder]): builder to delegate to
  """

  yaml_tag = '!ResidualLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, input_dim=512, layers=1, hidden_dim=Ref("exp_global.default_layer_dim"),
               residual_to_output=False, dropout=0.0, bidirectional=True, builder=None,
               yaml_path=None, decoder_input_dim=Ref("exp_global.default_layer_dim", default=None), decoder_input_feeding=True):
    self._final_states = None
    if yaml_path is not None and "decoder" in yaml_path:
      bidirectional = False
      if decoder_input_feeding:
        input_dim += decoder_input_dim
    if bidirectional:
      self.builder = self.add_serializable_component("builder", builder,
                                                     lambda: ResidualBiRNNBuilder(num_layers=layers,
                                                                                  input_dim=input_dim,
                                                                                  hidden_dim=hidden_dim,
                                                                                  add_to_output=residual_to_output,
                                                                                  dropout=dropout))
    else:
      self.builder = self.add_serializable_component("builder", builder,
                                                     lambda: ResidualRNNBuilder(num_layers=layers, input_dim=input_dim,
                                                                                hidden_dim=hidden_dim,
                                                                                add_to_output=residual_to_output,
                                                                                dropout=dropout))

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def __call__(self, sent):
    output = self.builder.transduce(sent)
    if not isinstance(output, ExpressionSequence):
      output = ExpressionSequence(expr_list=output)
    self._final_states = self.builder.get_final_states()
    return output

  def initial_state(self):
    return self.builder.initial_state()

  def get_final_states(self):
    assert self._final_states is not None, "ResidualLSTMSeqTransducer.__call__() must be invoked before ResidualLSTMSeqTransducer.get_final_states()"
    return self._final_states

class ResidualRNNBuilder(Serializable):
  """

  Builder for RNNs that implements additional residual connections between layers: the output of each
  intermediate hidden layer is added to its output.

  input ---> hidden layer 1 ---> hidden layer 2 -+--> ... -+---> hidden layer n ---+--->
                              \_________________/  \_ ... _/ \_(if add_to_output)_/

  Note: This is currently not serializable and therefore can't be directly specified in the config file; use :class:`xnmt.residual.ResidualLSTMSeqTransducer` instead.

  Args:
    num_layers (int): depth of the RNN (> 0)
    input_dim (int): size of the inputs
    hidden_dim (int): size of the outputs (and intermediate layer representations)
    add_to_output (bool): whether to add a residual connection to the output layer
    dropout (float): dropout probability; if None, use exp_global.dropout
    builder_layers (List[UniLSTMSeqTransducer]): builder layers
  """
  yaml_tag = "!ResidualRNNBuilder"
  @serializable_init
  def __init__(self, num_layers, input_dim, hidden_dim, add_to_output=False, dropout=None, builder_layers=None):
    assert num_layers > 0
    self.builder_layers = self.add_serializable_component("builder_layers", builder_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim, dropout=dropout) for i
      in range(num_layers)])

    self.add_to_output = add_to_output

  def whoami(self):
    return "ResidualRNNBuilder"

  def get_final_states(self):
    return self._final_states

  @staticmethod
  def _sum_lists(a, b):
    """
    Sums lists element-wise.
    """
    return [ea + eb for (ea, eb) in zip(a, b)]

  def add_inputs(self, es):
    """
    Returns the list of RNNStates obtained by adding the inputs to the RNN.

    Args:
      es: a list of Expression

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

    Args:
      es: a list of Expression

    see also add_inputs(xs), including for explanation of differences between
    add_inputs and this function.
    """
    es = self.builder_layers[0](es)
    self._final_states = [self.builder_layers[0].get_final_states()[0]]

    if len(self.builder_layers) == 1:
      return es

    for l in self.builder_layers[1:]:
      es = ExpressionSequence(expr_list=self._sum_lists(l(es), es))
      self._final_states.append(FinalTransducerState(es[-1], l.get_final_states()[0].cell_expr()))

    last_output = self.builder_layers[-1](es)

    if self.add_to_output:
      self._final_states.append(FinalTransducerState(last_output[-1], self.builder_layers[-1].get_final_states()[0].cell_expr()))
      return ExpressionSequence(expr_list=self._sum_lists(last_output, es))
    else:
      self._final_states.append(self.builder_layers[-1].get_final_states()[0])
      return last_output

  def initial_state(self):
    return PseudoState(self)


class ResidualBiRNNBuilder(Serializable):
  """
  A residual network with bidirectional first layer.

  Note: This is currently not serializable and therefore can't be directly specified in the config file; use :class:`xnmt.residual.ResidualLSTMSeqTransducer` instead.

  Args:
    num_layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    add_to_output (bool): whether to add a residual connection to the output layer
    dropout (float): dropout probability
    forward_layer (UniLSTMSeqTransducer):
    backward_layer (UniLSTMSeqTransducer):
    residual_network (ResidualRNNBuilder):
  """
  @serializable_init
  def __init__(self, num_layers, input_dim, hidden_dim, add_to_output=False,
               dropout=Ref("exp_global.dropout", default=0.0), forward_layer=None, backward_layer=None,
               residual_network=None):
    assert num_layers > 1
    assert hidden_dim % 2 == 0
    self.forward_layer = self.add_serializable_component("forward_layer", forward_layer,
                                                         lambda: UniLSTMSeqTransducer(input_dim=input_dim,
                                                                                      hidden_dim=hidden_dim / 2,
                                                                                      dropout=dropout))
    self.backward_layer = self.add_serializable_component("backward_layer", backward_layer,
                                                          lambda: UniLSTMSeqTransducer(input_dim=input_dim,
                                                                                       hidden_dim=hidden_dim / 2,
                                                                                       dropout=dropout))
    self.residual_network = self.add_serializable_component("residual_network", residual_network,
                                                            lambda: ResidualRNNBuilder(num_layers=num_layers - 1,
                                                                                       input_dim=hidden_dim,
                                                                                       hidden_dim=hidden_dim,
                                                                                       add_to_output=add_to_output,
                                                                                       dropout=dropout))

  def get_final_states(self):
    return self._final_states

  def add_inputs(self, es):
    return PseudoState(self, self.transduce(es))

  def transduce(self, es):
    forward_e = self.forward_layer(es)
    backward_e = self.backward_layer(ReversedExpressionSequence(es))
    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layer.get_final_states()[0].main_expr(),
                                                            self.backward_layer.get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layer.get_final_states()[0].cell_expr(),
                                                            self.backward_layer.get_final_states()[0].cell_expr()]))]

    output = self.residual_network.transduce(ExpressionSequence(expr_list=[dy.concatenate([f,b]) for f,b in zip(forward_e, ReversedExpressionSequence(backward_e))]))
    self._final_states += self.residual_network.get_final_states()
    return output

  def initial_state(self):
    return PseudoState(self)
