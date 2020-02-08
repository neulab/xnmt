import numbers
import collections.abc
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import xnmt
from xnmt import expression_seqs, param_collections, param_initializers
from xnmt import tensor_tools as tt
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init, Path

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch
  import torch.nn as nn

class UniLSTMState(object):
  """
  State object for UniLSTMSeqTransducer.
  """
  def __init__(self,
               network: 'UniLSTMSeqTransducer',
               prev: Optional['UniLSTMState'] = None,
               c: Sequence[tt.Tensor] = None,
               h: Sequence[tt.Tensor] = None) -> None:
    self._network = network
    if c is None:
      c = [tt.zeroes(hidden_dim=network.hidden_dim) for _ in range(network.num_layers)]
    if h is None:
      h = [tt.zeroes(hidden_dim=network.hidden_dim) for _ in range(network.num_layers)]
    self._c = tuple(c)
    self._h = tuple(h)
    self._prev = prev

  def add_input(self, x: Union[tt.Tensor, Sequence[tt.Tensor]]):
    new_c, new_h = self._network.add_input_to_prev(self, x)
    return UniLSTMState(self._network, prev=self, c=new_c, h=new_h)

  def b(self) -> 'UniLSTMSeqTransducer':
    return self._network

  def h(self) -> Sequence[tt.Tensor]:
    return self._h

  def c(self) -> Sequence[tt.Tensor]:
    return self._c

  def s(self) -> Sequence[tt.Tensor]:
    return self._c + self._h

  def prev(self) -> 'UniLSTMState':
    return self._prev

  def set_h(self, es: Optional[Sequence[tt.Tensor]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == self._network.num_layers
    self._h = tuple(es)
    return self

  def set_s(self, es: Optional[Sequence[tt.Tensor]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == 2 * self._network.num_layers
    self._c = tuple(es[:self._network.num_layers])
    self._h = tuple(es[self._network.num_layers:])
    return self

  def output(self) -> tt.Tensor:
    return self._h[-1]

  def __getitem__(self, item):
    return UniLSTMState(network=self._network,
                        prev=self._prev,
                        c=[ci[item] for ci in self._c],
                        h=[hi[item] for hi in self._h])

@xnmt.require_dynet
class UniLSTMSeqTransducerDynet(transducers.SeqTransducer, Serializable):
  """
  This implements a single LSTM layer based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.

  Args:
    layers: number of layers
    input_dim: input dimension
    hidden_dim: hidden dimension
    var_dropout: dropout probability (variational recurrent + vertical dropout)
    weightnoise_std: weight noise standard deviation
    param_init: how to initialize weight matrices. Position-specific initializers are ordered Wx_l0, Wh_l0, Wx_l1, Wh_l1, ...
    bias_init: how to initialize bias vectors. Position-specific initializers are ordered l0, l1, l2, ...
    yaml_path:
    decoder_input_dim: input dimension of the decoder; if ``yaml_path`` contains 'decoder' and ``decoder_input_feeding`` is True, this will be added to ``input_dim``
    decoder_input_feeding: whether this transducer is part of an input-feeding decoder; cf. ``decoder_input_dim``
  """
  yaml_tag = '!UniLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               var_dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               yaml_path: Path = Path(),
               decoder_input_dim: Optional[numbers.Integral] = Ref("exp_global.default_layer_dim", default=None),
               decoder_input_feeding: bool = True) -> None:
    self.num_layers = layers
    my_params = param_collections.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.dropout_rate = var_dropout
    self.weightnoise_std = weightnoise_std
    self.input_dim = input_dim
    self.total_input_dim = input_dim
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        self.total_input_dim += decoder_input_dim

    # [i; f; o; g]
    self.p_Wx = [my_params.add_parameters(dim=(hidden_dim*4, self.total_input_dim),
                                          init=param_init[0].initializer((hidden_dim*4, self.total_input_dim), num_shared=4))]
    self.p_Wx += [my_params.add_parameters(dim=(hidden_dim*4, hidden_dim),
                                           init=param_init[i*2].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(1, layers)]
    self.p_Wh = [my_params.add_parameters(dim=(hidden_dim*4, hidden_dim),
                                          init=param_init[i*2+1].initializer((hidden_dim*4, hidden_dim), num_shared=4)) for i in range(layers)]
    self.p_b  = [my_params.add_parameters(dim=(hidden_dim*4,),
                                          init=bias_init[i].initializer((hidden_dim*4,), num_shared=4)) for i in range(layers)]

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
      self.dropout_mask_x = [dy.random_bernoulli((self.total_input_dim,), retention_rate, scale, batch_size=batch_size)]
      self.dropout_mask_x += [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(1, self.num_layers)]
      self.dropout_mask_h = [dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size) for _ in range(self.num_layers)]

  def add_input_to_prev(self, prev_state: UniLSTMState, x: Union[tt.Tensor, Sequence[tt.Tensor]]) \
          -> Tuple[Sequence[tt.Tensor]]:
    if isinstance(x, dy.Expression):
      x = [x]
    elif type(x) != list:
      x = list(x)

    if self.dropout_rate > 0.0 and self.train and self.dropout_mask_x is None:
      self.set_dropout_masks(batch_size=tt.batch_size(x[0]))

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
    batch_size = expr_seq[0].batch_size()
    seq_len = expr_seq[0].sent_len()

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
        if (layer_i == 0 and sum([x_t_i.dim()[0][0] for x_t_i in x_t]) != self.total_input_dim) \
                or (layer_i>0 and sum([x_t_i.dim()[0][0] for x_t_i in x_t]) != self.hidden_dim):
          found_dim = sum([x_t_i.dim()[0][0] for x_t_i in x_t])
          raise ValueError(f"VanillaLSTMGates: x_t has inconsistent dimension {found_dim}, "
                           f"expecting {self.total_input_dim if layer_i==0 else self.hidden_dim}")
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

@xnmt.require_torch
class UniLSTMSeqTransducerTorch(transducers.SeqTransducer, Serializable):
  """
  This is a unidirecitonal LSTM that loops over time steps.

  Args:
    layers: number of layers
    input_dim: input dimension
    hidden_dim: hidden dimension
    var_dropout: dropout probability (variational recurrent + vertical dropout)
    param_init: how to initialize weight matrices. Position-specific initializers are ordered Wx_l0, Wh_l0, Wx_l1, Wh_l1, ...
    bias_init: how to initialize bias vectors. Must be a zero initializer.
    yaml_path:
    decoder_input_dim: input dimension of the decoder; if ``yaml_path`` contains 'decoder' and ``decoder_input_feeding`` is True, this will be added to ``input_dim``
    decoder_input_feeding: whether this transducer is part of an input-feeding decoder; cf. ``decoder_input_dim``
  """
  yaml_tag = '!UniLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               var_dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               weightnoise_std: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               yaml_path: Path = Path(),
               decoder_input_dim: Optional[numbers.Integral] = Ref("exp_global.default_layer_dim", default=None),
               decoder_input_feeding: bool = True) -> None:
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = var_dropout
    self.weightnoise_std = weightnoise_std
    self.input_dim = input_dim
    self.total_input_dim = input_dim
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        self.total_input_dim += decoder_input_dim

    my_params = param_collections.ParamManager.my_params(self)
    self.layers = nn.ModuleList([
      nn.LSTMCell(
        input_size=self.total_input_dim if layer == 0 else hidden_dim,
        hidden_size=hidden_dim,
      )
      for layer in range(layers)
    ]).to(xnmt.device)
    my_params.append(self.layers)
    my_params.init_params(param_init, bias_init)
    # init forget gate biases to 1
    for name, param in self.layers.named_parameters():
      if 'bias_ih' in name:
        # Pytorch using redundant biases 'bias_ih' and 'bias_hh'. Initializing only one to 1, the other one to zero:
        n = param.size(0)
        start, end = n // 4, n // 2
        param.data[start:end].fill_(1)
      if 'bias_hh' in name:
        # Don't update params for the unused redundant bias. This ensures consistency with DyNet LSTM implementation.
        param.requires_grad = False

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
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

      self.dropout_mask_x = [torch.autograd.Variable(torch.bernoulli(torch.empty((batch_size, self.total_input_dim), device=xnmt.device).fill_(retention_rate))) * scale]
      self.dropout_mask_x += [torch.autograd.Variable(torch.bernoulli(torch.empty((batch_size, self.hidden_dim), device=xnmt.device).fill_(retention_rate))) * scale for _ in range(1, self.num_layers)]
      self.dropout_mask_h = [torch.autograd.Variable(torch.bernoulli(torch.empty((batch_size, self.hidden_dim), device=xnmt.device).fill_(retention_rate))) * scale for _ in range(self.num_layers)]

  def add_input_to_prev(self, prev_state: UniLSTMState, x: tt.Tensor) \
          -> Tuple[Sequence[tt.Tensor]]:
    assert isinstance(x, tt.Tensor)

    if self.dropout_rate > 0.0 and self.train and self.dropout_mask_x is None:
      self.set_dropout_masks(batch_size=tt.batch_size(x))

    new_c, new_h = [], []
    for layer_i in range(self.num_layers):
      h_tm1 = prev_state._h[layer_i]
      if self.dropout_rate > 0.0 and self.train:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        x = torch.mul(x, self.dropout_mask_x[layer_i])
        h_tm1 = torch.mul(h_tm1, self.dropout_mask_h[layer_i])
      h_t, c_t = self.layers[layer_i](x, (h_tm1, prev_state._c[layer_i]))
      new_c.append(c_t)
      new_h.append(h_t)
      x = h_t

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
    concat_inputs = len(expr_seq)>=2
    batch_size = tt.batch_size(expr_seq[0][0])
    seq_len = expr_seq[0].sent_len()
    mask = expr_seq[0].mask

    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    cur_input = expr_seq
    self._final_states = []
    for layer_i in range(self.num_layers):
      h = [tt.zeroes(hidden_dim=self.hidden_dim, batch_size=batch_size)]
      c = [tt.zeroes(hidden_dim=self.hidden_dim, batch_size=batch_size)]
      for pos_i in range(seq_len):
        if concat_inputs and layer_i==0:
          x_t = tt.concatenate([cur_input[i][pos_i] for i in range(len(cur_input))])
        else:
          x_t = cur_input[0][pos_i]
        h_tm1 = h[-1]
        if self.dropout_rate > 0.0 and self.train:
          # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
          x_t = torch.mul(x_t, self.dropout_mask_x[layer_i])
          h_tm1 = torch.mul(h_tm1, self.dropout_mask_h[layer_i])
        h_t, c_t = self.layers[layer_i](x_t, (h_tm1, c[-1]))
        if mask is None or np.isclose(np.sum(mask.np_arr[:,pos_i:pos_i+1]), 0.0):
          c.append(c_t)
          h.append(h_t)
        else:
          c.append(mask.cmult_by_timestep_expr(c_t,pos_i,True) + mask.cmult_by_timestep_expr(c[-1],pos_i,False))
          h.append(mask.cmult_by_timestep_expr(h_t,pos_i,True) + mask.cmult_by_timestep_expr(h[-1],pos_i,False))
      self._final_states.append(transducers.FinalTransducerState(h[-1], c[-1]))
      cur_input = [h[1:]]

    return expression_seqs.ExpressionSequence(expr_list=h[1:], mask=mask)

  def params_from_dynet(self, arrays, state_dict):
    assert len(arrays)==3
    h_dim = arrays[0].shape[0] // 4
    arrays[0] = np.concatenate([arrays[0][:h_dim * 2,:], arrays[0][h_dim * 3:,:],
                         arrays[0][h_dim * 2:h_dim * 3, :]], axis=0)
    arrays[1] = np.concatenate([arrays[1][:h_dim * 2,:], arrays[1][h_dim * 3:,:],
                         arrays[1][h_dim * 2:h_dim * 3,:]], axis=0)
    arrays[2] = np.concatenate([arrays[2][:h_dim], arrays[2][h_dim:h_dim * 2] + 1,
                         arrays[2][h_dim * 3:], arrays[2][h_dim * 2:h_dim * 3]], axis=0)

    return {'0.0.weight_ih':arrays[0],
            '0.0.weight_hh':arrays[1],
            '0.0.bias_ih':arrays[2],
            '0.0.bias_hh':np.zeros_like(arrays[2])}

UniLSTMSeqTransducer = xnmt.resolve_backend(UniLSTMSeqTransducerDynet, UniLSTMSeqTransducerTorch)

class BiLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than DyNet's CompactVanillaLSTMBuilder due to avoiding concat operations.
  It uses 2 :class:`xnmt.lstm.UniLSTMSeqTransducer` objects in each layer.

  Args:
    layers: number of layers
    input_dim: input dimension
    hidden_dim: hidden dimension
    var_dropout: dropout probability (variational recurrent + vertical dropout)
    param_init: how to initialize weight matrices. In case of an InitializerSequence, the order is fwd_l0, bwd_l0, fwd_l1, bwd_l1, ..
    bias_init: how to initialize bias vectors. In case of an InitializerSequence, the order is fwd_l0, bwd_l0, fwd_l1, bwd_l1, ..
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
               var_dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               forward_layers : Optional[Sequence[UniLSTMSeqTransducer]] = None,
               backward_layers: Optional[Sequence[UniLSTMSeqTransducer]] = None) -> None:
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = var_dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers",
                                                          forward_layers,
                                                          lambda: [UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim,
                                                                                        hidden_dim=hidden_dim // 2,
                                                                                        var_dropout=var_dropout,
                                                                                        param_init=param_init[i*2],
                                                                                        bias_init=bias_init[i*2])
                                                                   for i in range(layers)])
    self.backward_layers = self.add_serializable_component("backward_layers",
                                                           backward_layers,
                                                           lambda: [UniLSTMSeqTransducer(input_dim=input_dim if i == 0 else hidden_dim,
                                                                                         hidden_dim=hidden_dim // 2,
                                                                                         var_dropout=var_dropout,
                                                                                         param_init=param_init[i*2+1],
                                                                                         bias_init=bias_init[i*2+1])
                                                                    for i in range(layers)])

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
      transducers.FinalTransducerState(tt.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                       tt.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return expression_seqs.ExpressionSequence(expr_list=[tt.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(forward_es.sent_len())], mask=mask)

@xnmt.require_dynet
class CustomLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.

  Args:
    layers: number of layers
    input_dim: input dimension; if None, use exp_global.default_layer_dim
    hidden_dim: hidden dimension; if None, use exp_global.default_layer_dim
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
    my_params = param_collections.ParamManager.my_params(self)

    # [i; f; o; g]
    self.p_Wx = my_params.add_parameters(dim=(hidden_dim*4, input_dim), init=param_init.initializer((hidden_dim*4, input_dim)))
    self.p_Wh = my_params.add_parameters(dim=(hidden_dim*4, hidden_dim), init=param_init.initializer((hidden_dim*4, hidden_dim)))
    self.p_b  = my_params.add_parameters(dim=(hidden_dim*4,), init=bias_init.initializer((hidden_dim*4,)))

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

@xnmt.require_torch
class CudnnLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  An LSTM using CuDNN acceleration, potentially stacked and bidirectional.

  Because CuDNN is used, only basic (vertical) dropout is supported.

  Args:
    layers: number of layers
    bidirectional: whether two use uni- or bidirectional LSTM
    input_dim: input dimension
    hidden_dim: hidden dimension
    vert_dropout: dropout probability (on outputs only)
    param_init: how to initialize weight matrices. In case of an InitializerSequence, the order is fwd_l0_ih, fwd_l0_hh, fwd_l1_ih, ... for unidirectional and fwd_l0_ih, fwd_l0_hh, bwd_l0_ih, bwd_l0_hh, fwd_l1_ih, .. for bidirectional
    bias_init: how to initialize bias vectors. In case of an InitializerSequence, the order is fwd_l0, fwd_l1, ... for unidirectional and fwd_l0, bwd_l0, fwd_l1, bwd_l1, .. for bidirectional
  """
  yaml_tag = '!CudnnLSTMSeqTransducer'

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               bidirectional: bool = False,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               vert_dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ZeroInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))) -> None:
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    if bidirectional: assert hidden_dim % 2 == 0
    self.num_dir = 2 if bidirectional else 1

    self.lstm = nn.LSTM(input_size=input_dim,
                        hidden_size=self.hidden_dim//self.num_dir,
                        num_layers=self.num_layers,
                        bidirectional=bidirectional,
                        dropout=vert_dropout if self.num_layers>1 else 0, # avoid pytorch warning
                        batch_first=True).to(xnmt.device)
    my_params = param_collections.ParamManager.my_params(self)
    my_params.append(self.lstm)
    my_params.init_params(param_init, bias_init)

    # init forget gate biases to 1
    for name, param in self.lstm.named_parameters():
      if 'bias_ih' in name:
        # Pytorch using redundant biases 'bias_ih' and 'bias_hh'. Initializing only one to 1, the other one to zero:
        n = param.size(0)
        start, end = n // 4, n // 2
        param.data[start:end].fill_(1)
      if 'bias_hh' in name:
        # Don't update params for the unused redundant bias. This ensures consistency with DyNet LSTM implementation.
        param.requires_grad = False



  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.lstm.train(mode=val)
    self.train = val

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, es: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':

    batch_size = tt.batch_size(es.as_tensor())
    if es.mask:
      seq_lengths = es.mask.seq_lengths()
    else:
      seq_lengths = [es.sent_len()] * batch_size

    # Sort the input and lengths as the descending order
    seq_lengths = torch.LongTensor(seq_lengths).to(xnmt.device)
    lengths, perm_index = seq_lengths.sort(0, descending=True)
    sorted_input = es.as_tensor()[perm_index]

    perm_index_rev = [-1] * len(lengths)
    for i in range(len(lengths)):
      perm_index_rev[perm_index[i]] = i
    perm_index_rev = torch.LongTensor(perm_index_rev).to(xnmt.device)

    packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input, list(lengths.data), batch_first=True)
    state_size = self.num_dir * self.num_layers, batch_size, self.hidden_dim // self.num_dir
    h0 = sorted_input.new_zeros(*state_size)
    c0 = sorted_input.new_zeros(*state_size)
    output, (final_hiddens, final_cells) = self.lstm(packed_input, (h0, c0))
    output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=es.sent_len())[0]

    # restore the sorting
    decoded = output[perm_index_rev]

    self._final_states = []
    for layer_i in range(self.num_layers):
      final_hidden = final_hiddens.view(self.num_layers,
                                        self.num_dir,
                                        batch_size, -1)[layer_i].transpose(0, 1).contiguous().view(batch_size, -1)
      final_hidden = final_hidden[perm_index_rev]
      self._final_states.append(transducers.FinalTransducerState(final_hidden))

    ret = expression_seqs.ExpressionSequence(expr_tensor=decoded, mask=es.mask)
    return ret

