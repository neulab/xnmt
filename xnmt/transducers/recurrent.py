import numbers
import collections.abc
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import dynet as dy

from xnmt import expression_seqs, param_collections, param_initializers
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init, Path

class UniLSTMState(object):
  """
  State object for UniLSTMSeqTransducer.
  """
  def __init__(self,
               network: 'UniLSTMSeqTransducer',
               prev: Optional['UniLSTMState'] = None,
               c: Sequence[dy.Expression] = None,
               h: Sequence[dy.Expression] = None) -> None:
    self._network = network
    if c is None:
      c = [dy.zeroes(dim=(network.hidden_dim,)) for _ in range(network.num_layers)]
    if h is None:
      h = [dy.zeroes(dim=(network.hidden_dim,)) for _ in range(network.num_layers)]
    self._c = tuple(c)
    self._h = tuple(h)
    self._prev = prev

  def add_input(self, x: Union[dy.Expression, Sequence[dy.Expression]]):
    new_c, new_h = self._network.add_input_to_prev(self, x)
    return UniLSTMState(self._network, prev=self, c=new_c, h=new_h)

  def b(self) -> 'UniLSTMSeqTransducer':
    return self._network

  def h(self) -> Sequence[dy.Expression]:
    return self._h

  def s(self) -> Sequence[dy.Expression]:
    return self._c + self._h

  def prev(self) -> 'UniLSTMState':
    return self._prev

  def set_h(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == self._network.num_layers
    self._h = tuple(es)
    return self

  def set_s(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == 2 * self._network.num_layers
    self._c = tuple(es[:self._network.num_layers])
    self._h = tuple(es[self._network.num_layers:])
    return self

  def output(self) -> dy.Expression:
    return self._h[-1]

  def __getitem__(self, item):
    return UniLSTMState(network=self._network,
                        prev=self._prev,
                        c=[ci[item] for ci in self._c],
                        h=[hi[item] for hi in self._h])


class UniLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a single LSTM layer based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
    yaml_path (str):
    decoder_input_dim (int): input dimension of the decoder; if ``yaml_path`` contains 'decoder' and ``decoder_input_feeding`` is True, this will be added to ``input_dim``
    decoder_input_feeding (bool): whether this transducer is part of an input-feeding decoder; cf. ``decoder_input_dim``
  """
  yaml_tag = '!UniLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               yaml_path: Path = Path(),
               decoder_input_dim: Optional[numbers.Integral] = Ref("exp_global.default_layer_dim", default=None),
               decoder_input_feeding: bool = True) -> None:
    self.num_layers = layers
    model = param_collections.ParamManager.my_params(self)
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        input_dim += decoder_input_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    self.input_dim = input_dim

    if not isinstance(param_init, collections.abc.Sequence):
      param_init = [param_init] * layers
    if not isinstance(bias_init, collections.abc.Sequence):
        bias_init = [bias_init] * layers

    # [i; f; o; g]
    self.p_Wx = [model.add_parameters(dim=(hidden_dim*4, input_dim), init=param_init[0].initializer((hidden_dim*4, input_dim), num_shared=4))]
    self.p_Wx += [model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init[i].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(1, layers)]
    self.p_Wh = [model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init[i].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(layers)]
    self.p_b  = [model.add_parameters(dim=(hidden_dim*4,), init=bias_init[i].initializer((hidden_dim*4,), num_shared=4)) for i in range(layers)]

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.Wx = [dy.parameter(Wx) for Wx in self.p_Wx]
    self.Wh = [dy.parameter(Wh) for Wh in self.p_Wh]
    self.b = [dy.parameter(b) for b in self.p_b]
    self.dropout_mask_x = None
    self.dropout_mask_h = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def initial_state(self) -> UniLSTMState:
    return UniLSTMState(self)

  def set_dropout(self, dropout: numbers.Real) -> None:
    self.dropout_rate = dropout

  def set_dropout_masks(self, batch_size: numbers.Integral = 1) -> None:
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = [dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)]
      self.dropout_mask_x += [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(1, self.num_layers)]
      self.dropout_mask_h = [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(self.num_layers)]

  def add_input_to_prev(self, prev_state: UniLSTMState, x: Union[dy.Expression, Sequence[dy.Expression]]) \
          -> Tuple[Sequence[dy.Expression]]:
    if isinstance(x, dy.Expression):
      x = [x]
    elif type(x) != list:
      x = list(x)

    if self.dropout_rate > 0.0 and self.train and self.dropout_mask_x is None:
      self.set_dropout_masks()

    new_c, new_h = [], []
    for layer_i in range(self.num_layers):
      if self.dropout_rate > 0.0 and self.train:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates = dy.vanilla_lstm_gates_dropout_concat(
          x, prev_state._h[layer_i], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i],
          self.dropout_mask_x[layer_i], self.dropout_mask_h[layer_i],
          self.weightnoise_std if self.train else 0.0)
      else:
        gates = dy.vanilla_lstm_gates_concat(
          x, prev_state._h[layer_i], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i],
          self.weightnoise_std if self.train else 0.0)
      new_c.append(dy.vanilla_lstm_c(prev_state._c[layer_i], gates))
      new_h.append(dy.vanilla_lstm_h(new_c[-1], gates))
      x = [new_h[-1]]

    return new_c, new_h

  def transduce(self, expr_seq: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """
    if isinstance(expr_seq, expression_seqs.ExpressionSequence):
      expr_seq = [expr_seq]
    batch_size = expr_seq[0][0].dim()[1]
    seq_len = len(expr_seq[0])

    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    cur_input = expr_seq
    self._final_states = []
    for layer_i in range(self.num_layers):
      h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
      c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
      for pos_i in range(seq_len):
        x_t = [cur_input[j][pos_i] for j in range(len(cur_input))]
        if isinstance(x_t, dy.Expression):
          x_t = [x_t]
        elif type(x_t) != list:
          x_t = list(x_t)
        if sum([x_t_i.dim()[0][0] for x_t_i in x_t]) != self.input_dim:
          found_dim = sum([x_t_i.dim()[0][0] for x_t_i in x_t])
          raise ValueError(f"VanillaLSTMGates: x_t has inconsistent dimension {found_dim}, expecting {self.input_dim}")
        if self.dropout_rate > 0.0 and self.train:
          # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
          gates_t = dy.vanilla_lstm_gates_dropout_concat(x_t,
                                                         h[-1],
                                                         self.Wx[layer_i],
                                                         self.Wh[layer_i],
                                                         self.b[layer_i],
                                                         self.dropout_mask_x[layer_i],
                                                         self.dropout_mask_h[layer_i],
                                                         self.weightnoise_std if self.train else 0.0)
        else:
          gates_t = dy.vanilla_lstm_gates_concat(x_t, h[-1], self.Wx[layer_i], self.Wh[layer_i], self.b[layer_i], self.weightnoise_std if self.train else 0.0)
        c_t = dy.vanilla_lstm_c(c[-1], gates_t)
        h_t = dy.vanilla_lstm_h(c_t, gates_t)
        if expr_seq[0].mask is None or np.isclose(np.sum(expr_seq[0].mask.np_arr[:,pos_i:pos_i+1]), 0.0):
          c.append(c_t)
          h.append(h_t)
        else:
          c.append(expr_seq[0].mask.cmult_by_timestep_expr(c_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(c[-1],pos_i,False))
          h.append(expr_seq[0].mask.cmult_by_timestep_expr(h_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(h[-1],pos_i,False))
      self._final_states.append(transducers.FinalTransducerState(h[-1], c[-1]))
      cur_input = [h[1:]]

    return expression_seqs.ExpressionSequence(expr_list=h[1:], mask=expr_seq[0].mask)

class BiLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than DyNet's CompactVanillaLSTMBuilder due to avoiding concat operations.
  It uses 2 :class:`xnmt.lstm.UniLSTMSeqTransducer` objects in each layer.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects 
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
    forward_layers: set automatically
    backward_layers: set automatically
  """
  yaml_tag = '!BiLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               forward_layers : Optional[Sequence[UniLSTMSeqTransducer]] = None,
               backward_layers: Optional[Sequence[UniLSTMSeqTransducer]] = None) -> None:
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers", forward_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, collections.abc.Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, collections.abc.Sequence) else bias_init) for i in
      range(layers)])
    self.backward_layers = self.add_serializable_component("backward_layers", backward_layers, lambda: [
      UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim, hidden_dim=hidden_dim // 2, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init[i] if isinstance(param_init, collections.abc.Sequence) else param_init,
                           bias_init=bias_init[i] if isinstance(bias_init, collections.abc.Sequence) else bias_init) for i in
      range(layers)])

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, es: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    mask = es.mask
     # first layer
    forward_es = self.forward_layers[0].transduce(es)
    rev_backward_es = self.backward_layers[0].transduce(expression_seqs.ReversedExpressionSequence(es))

    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i].transduce([forward_es, expression_seqs.ReversedExpressionSequence(rev_backward_es)])
      rev_backward_es = expression_seqs.ExpressionSequence(
        self.backward_layers[layer_i].transduce([expression_seqs.ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(),
        mask=mask)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return expression_seqs.ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask)


class CustomLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension; if None, use exp_global.default_layer_dim
    hidden_dim (int): hidden dimension; if None, use exp_global.default_layer_dim
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
                If None, use ``exp_global.param_init``
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
               If None, use ``exp_global.param_init``
  """
  yaml_tag = "!CustomLSTMSeqTransducer"

  @serializable_init
  def __init__(self,
               layers: numbers.Integral,
               input_dim: numbers.Integral,
               hidden_dim: numbers.Integral,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    if layers!=1: raise RuntimeError("CustomLSTMSeqTransducer supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = param_collections.ParamManager.my_params(self)

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim), init=param_init.initializer((hidden_dim*4, input_dim)))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init.initializer((hidden_dim*4, hidden_dim)))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=bias_init.initializer((hidden_dim*4,)))

  def transduce(self, xs: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
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

