from __future__ import division, generators

import dynet as dy
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState


class LSTMSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = u'!LSTMSeqTransducer'

  def __init__(self, yaml_context, input_dim=None, layers=1, hidden_dim=None, dropout=None, bidirectional=True):
    register_handler(self)
    model = yaml_context.dynet_param_collection.param_col
    input_dim = input_dim or yaml_context.default_layer_dim
    hidden_dim = hidden_dim or yaml_context.default_layer_dim
    dropout = dropout or yaml_context.dropout
    self.input_dim = input_dim
    self.layers = layers
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    if bidirectional:
      self.builder = BiCompactLSTMBuilder(layers, input_dim, hidden_dim, model)
    else:
      self.builder = CustomCompactLSTMBuilder(layers, input_dim, hidden_dim, model)

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
    if hasattr(self.builder, "get_final_states"):
      self._final_states = self.builder.get_final_states()
    else:
      self._final_states = [FinalTransducerState(output[-1])]
    return output

  def get_final_states(self):
    assert self._final_states is not None, "LSTMSeqTransducer.__call__() must be invoked before LSTMSeqTransducer.get_final_states()"
    return self._final_states


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


class LSTMState(object):
  def __init__(self, builder, h_t=None, c_t=None, state_idx=-1, prev_state=None):
    self.builder = builder
    self.state_idx=state_idx
    self.prev_state = prev_state
    self.h_t = h_t
    self.c_t = c_t

  def add_input(self, x_t):
    h_t, c_t = self.builder.add_input(x_t, self.prev_state)
    return LSTMState(self.builder, h_t, c_t, self.state_idx+1, prev_state=self)

  def transduce(self, xs):
    return self.builder.transduce(xs)

  def output(self): return self.h_t

  def prev(self): return self.prev_state
  def b(self): return self.builder
  def get_state_idx(self): return self.state_idx


class CustomCompactLSTMBuilder(object):
  """
  This implements an LSTM builder based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.
  """
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("CustomCompactLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))

    self.dropout_rate = 0.0
    self.weightnoise_std = 0.0

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  def get_final_states(self):
    return self._final_states

  def whoami(self): return "CustomCompactLSTMBuilder"

  def set_dropout(self, p):
    if not 0.0 <= p <= 1.0:
      raise Exception("dropout rate must be a probability (>=0 and <=1)")
    self.dropout_rate = p
  def disable_dropout(self):
    self.dropout_rate = 0.0

  def set_weightnoise(self, std):
    if not 0.0 <= std:
      raise Exception("weight noise must have standard deviation >=0")
    self.weightnoise_std = std
  def disable_weightnoise(self):
    self.weightnoise_std = 0.0

  def initial_state(self, vecs=None):
    self._final_states = None
    self.Wx = dy.parameter(self.p_Wx)
    self.Wh = dy.parameter(self.p_Wh)
    self.b = dy.parameter(self.p_b)
    self.dropout_mask_x = None
    self.dropout_mask_h = None
    if vecs is not None:
      assert len(vecs)==2
      return LSTMState(self, h_t=vecs[0], c_t=vecs[1])
    else:
      return LSTMState(self)
  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def add_input(self, x_t, prev_state):
    batch_size = x_t.dim()[1]
    if self.dropout_rate > 0.0 and (self.dropout_mask_x is None or self.dropout_mask_h is None):
      self.set_dropout_masks(batch_size=batch_size)
    if prev_state is None or prev_state.h_t is None:
      h_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)
    else:
      h_tm1 = prev_state.h_t
    if prev_state is None or prev_state.c_t is None:
      c_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=x_t.dim()[1])
    else:
      c_tm1 = prev_state.c_t
    if self.dropout_rate > 0.0:
      # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
      gates_t = dy.vanilla_lstm_gates_dropout(x_t, h_tm1, self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std)
    else:
      gates_t = dy.vanilla_lstm_gates(x_t, h_tm1, self.Wx, self.Wh, self.b, self.weightnoise_std)
    c_t = dy.vanilla_lstm_c(c_tm1, gates_t)
    h_t = dy.vanilla_lstm_h(c_t, gates_t)
    return h_t, c_t
    
  def transduce(self, expr_seq):
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)
    
    :param expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    :returns: expression sequence
    """
    self.initial_state()
    if isinstance(expr_seq, ExpressionSequence):
      expr_seq = [expr_seq]
    batch_size = expr_seq[0][0].dim()[1]
    seq_len = len(expr_seq[0])
    
    if self.dropout_rate > 0.0:
      self.set_dropout_masks(batch_size=batch_size)
      
    h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    for pos_i in range(seq_len):
      x_t = [expr_seq[j][pos_i] for j in range(len(expr_seq))]
      if isinstance(x_t, dy.Expression):
        x_t = [x_t]
      elif type(x_t) != list:
        x_t = list(x_t)
      if self.dropout_rate > 0.0:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates_t = dy.vanilla_lstm_gates_dropout_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std)
      else:
        gates_t = dy.vanilla_lstm_gates_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.weightnoise_std)
      c_t = dy.vanilla_lstm_c(c[-1], gates_t)
      h_t = dy.vanilla_lstm_h(c_t, gates_t)
      if expr_seq[0].mask is None:
        c.append(c_t)
        h.append(h_t)
      else:
        c.append(expr_seq[0].mask.cmult_by_timestep_expr(c_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(c[-1],pos_i,False))
        h.append(expr_seq[0].mask.cmult_by_timestep_expr(h_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(h[-1],pos_i,False))
    self._final_states = [FinalTransducerState(h[-1], c[-1])]
    return ExpressionSequence(expr_list=h[1:], mask=expr_seq[0].mask)

class BiCompactLSTMBuilder:
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than the native CompactVanillaLSTMBuilder due to avoiding concat operations.
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model):
    self.num_layers = num_layers
    assert hidden_dim % 2 == 0
    self.forward_layers = [CustomCompactLSTMBuilder(1, input_dim, hidden_dim/2, model)]
    self.backward_layers = [CustomCompactLSTMBuilder(1, input_dim, hidden_dim/2, model)]
    self.forward_layers += [CustomCompactLSTMBuilder(1, hidden_dim, hidden_dim/2, model) for _ in range(num_layers-1)]
    self.backward_layers += [CustomCompactLSTMBuilder(1, hidden_dim, hidden_dim/2, model) for _ in range(num_layers-1)]

  def get_final_states(self):
    return self._final_states

  def set_dropout(self, p):
    for layer in self.forward_layers + self.backward_layers:
      layer.set_dropout(p)

  def disable_dropout(self):
    for layer in self.forward_layers + self.backward_layers:
      layer.disable_dropout()

  def set_weightnoise(self, std):
    for layer in self.forward_layers + self.backward_layers:
      layer.set_weightnoise(std)

  def disable_weightnoise(self):
    for layer in self.forward_layers + self.backward_layers:
      layer.disable_weightnoise()

  def transduce(self, es):
    mask = es.mask
    # first layer
    forward_es = self.forward_layers[0].initial_state().transduce(es)
    rev_backward_es = self.backward_layers[0].initial_state().transduce(ReversedExpressionSequence(es))

    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i].initial_state().transduce([forward_es, ReversedExpressionSequence(rev_backward_es)])
      rev_backward_es = ExpressionSequence(self.backward_layers[layer_i].initial_state().transduce([ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(), mask=mask)
      forward_es = new_forward_es

    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
                          for layer_i in range(len(self.forward_layers))]
    return ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask)

  def initial_state(self):
    return PseudoState(self)


class CustomLSTMBuilder(object):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.
  """
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("CustomLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))

  def whoami(self): return "CustomLSTMBuilder"

  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("CustomLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def transduce(self, xs):
    Wx = dy.parameter(self.p_Wx)
    Wh = dy.parameter(self.p_Wh)
    b = dy.parameter(self.p_b)
    h = []
    c = []
    for i, x_t in enumerate(xs):
      if i==0:
        tmp = dy.affine_transform([b, Wx, x_t])
      else:
        tmp = dy.affine_transform([b, Wx, x_t, Wh, h[-1]])
      i_ait = dy.pick_range(tmp, 0, self.hidden_dim)
      i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pick_range(tmp, self.hidden_dim*2, self.hidden_dim*3)
      i_agt = dy.pick_range(tmp, self.hidden_dim*3, self.hidden_dim*4)
      i_it = dy.logistic(i_ait)
      i_ft = dy.logistic(i_aft + 1.0)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if i==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
      h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
    return h

