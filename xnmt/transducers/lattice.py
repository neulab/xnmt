from typing import List, Optional, Sequence
import numbers

import xnmt
from xnmt import events, expression_seqs, param_collections
from xnmt.transducers import base as transducers
from xnmt.persistence import Ref, Serializable, serializable_init

if xnmt.backend_dynet:
  import dynet as dy

@xnmt.require_dynet
class LatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A lattice LSTM.

  This is the unidirectional single-layer lattice LSTM, as described here:
    Sperber et al.: Neural Lattice-to-Sequence Models for Uncertain Inputs (EMNLP 2017)
    http://aclweb.org/anthology/D17-1145

  Note that lattice scores are currently not handled.

  Args:
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
  """

  yaml_tag = "!LatticeLSTMTransducer"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0)) -> None:
    self.dropout_rate = dropout
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    my_params = param_collections.ParamManager.my_params(self)

    # [i; o; g]
    self.p_Wx_iog = my_params.add_parameters(dim=(hidden_dim * 3, input_dim))
    self.p_Wh_iog = my_params.add_parameters(dim=(hidden_dim * 3, hidden_dim))
    self.p_b_iog = my_params.add_parameters(dim=(hidden_dim * 3,), init=dy.ConstInitializer(0.0))
    self.p_Wx_f = my_params.add_parameters(dim=(hidden_dim, input_dim))
    self.p_Wh_f = my_params.add_parameters(dim=(hidden_dim, hidden_dim))
    self.p_b_f = my_params.add_parameters(dim=(hidden_dim,), init=dy.ConstInitializer(1.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.dropout_mask_x = None
    self.dropout_mask_h = None
    self.cur_src = src

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def set_dropout_masks(self, batch_size: numbers.Integral = 1) -> None:
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def transduce(self, expr_seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    if expr_seq.dim()[1] > 1: raise ValueError(f"LatticeLSTMTransducer requires batch size 1, got {expr_seq.dim()[1]}")
    lattice = self.cur_src[0]
    Wx_iog = dy.parameter(self.p_Wx_iog)
    Wh_iog = dy.parameter(self.p_Wh_iog)
    b_iog = dy.parameter(self.p_b_iog)
    Wx_f = dy.parameter(self.p_Wx_f)
    Wh_f = dy.parameter(self.p_Wh_f)
    b_f = dy.parameter(self.p_b_f)
    h = {}
    c = {}
    h_list = []

    batch_size = expr_seq.dim()[1]
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    for i, cur_node_id in enumerate(lattice.nodes):
      prev_node = lattice.graph.predecessors(cur_node_id)
      val = expr_seq[i]
      if self.dropout_rate > 0.0 and self.train:
        val = dy.cmult(val, self.dropout_mask_x)
      i_ft_list = []
      if len(prev_node) == 0:
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val])
      else:
        h_tilde = sum(h[pred] for pred in prev_node)
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val, Wh_iog, h_tilde])
        for pred in prev_node:
          i_ft_list.append(dy.logistic(dy.affine_transform([b_f, Wx_f, val, Wh_f, h[pred]])))
      i_ait = dy.pick_range(tmp_iog, 0, self.hidden_dim)
      i_aot = dy.pick_range(tmp_iog, self.hidden_dim, self.hidden_dim * 2)
      i_agt = dy.pick_range(tmp_iog, self.hidden_dim * 2, self.hidden_dim * 3)

      i_it = dy.logistic(i_ait)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if len(prev_node) == 0:
        c[cur_node_id] = dy.cmult(i_it, i_gt)
      else:
        fc = dy.cmult(i_ft_list[0], c[prev_node[0]])
        for i in range(1, len(prev_node)):
          fc += dy.cmult(i_ft_list[i], c[prev_node[i]])
        c[cur_node_id] = fc + dy.cmult(i_it, i_gt)
      h_t = dy.cmult(i_ot, dy.tanh(c[cur_node_id]))
      if self.dropout_rate > 0.0 and self.train:
        h_t = dy.cmult(h_t, self.dropout_mask_h)
      h[cur_node_id] = h_t
      h_list.append(h_t)
    self._final_states = [transducers.FinalTransducerState(h_list[-1], h_list[-1])]
    return expression_seqs.ExpressionSequence(expr_list=h_list)

@xnmt.require_dynet
class BiLatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A multi-layered bidirectional lattice LSTM.

  Makes use of several LatticeLSTMTransducer instances and combines them appropriately.

  Args:
    layers: number of layers
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
    forward_layers: determined automatically
    backward_layers: determined automatically
  """

  yaml_tag = '!BiLatticeLSTMTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               forward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None,
               backward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None) -> None:
    self.num_layers = layers
    input_dim = input_dim
    hidden_dim = hidden_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers",
                                                          forward_layers,
                                                          lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                        hidden_dim=hidden_dim,
                                                                                        dropout=dropout,
                                                                                        layers=layers))
    self.backward_layers = self.add_serializable_component("backward_layers",
                                                           backward_layers,
                                                           lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                         hidden_dim=hidden_dim,
                                                                                         dropout=dropout,
                                                                                         layers=layers))

  def _make_dir_layers(self, input_dim, hidden_dim, dropout, layers):
    dir_layers = [LatticeLSTMTransducer(input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout)]
    dir_layers += [LatticeLSTMTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout) for
                            _ in range(layers - 1)]
    return dir_layers

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_src = src
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, expr_sequence: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    # first layer
    forward_es = self.forward_layers[0].transduce(expr_sequence)
    rev_backward_es = self.backward_layers[0].transduce(expression_seqs.ReversedExpressionSequence(expr_sequence))

    for layer_i in range(1, len(self.forward_layers)):
      concat_fwd = expression_seqs.ExpressionSequence(expr_list=[dy.concatenate([fwd_expr, bwd_expr])
                                                                 for fwd_expr, bwd_expr
                                                                 in zip(forward_es.as_list(),
                                                                        reversed(rev_backward_es.as_list()))])
      concat_bwd = expression_seqs.ExpressionSequence(expr_list=[dy.concatenate([fwd_expr, bwd_expr])
                                                                 for fwd_expr, bwd_expr
                                                                 in zip(reversed(forward_es.as_list()),
                                                                                 rev_backward_es.as_list())])
      new_forward_es = self.forward_layers[layer_i].transduce(concat_fwd)
      rev_backward_es = self.backward_layers[layer_i].transduce(concat_bwd)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return expression_seqs.ExpressionSequence(expr_list=[dy.concatenate([forward_es[i], rev_backward_es[-i - 1]])
                                                         for i in range(forward_es.sent_len())])
