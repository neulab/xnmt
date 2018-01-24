from __future__ import division, generators

import numpy as np
import dynet as dy

from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.serialize.serializable import Serializable
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.serialize.tree_tools import Ref, Path

class UniLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  This implements an LSTM builder based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.
  """
  yaml_tag = u'!UniLSTMSeqTransducer'
  
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), input_dim=None, hidden_dim=None, dropout = None, weightnoise_std=None):
    register_handler(self)
    model = xnmt_global.dynet_param_collection.param_col
    input_dim = input_dim or xnmt_global.default_layer_dim
    hidden_dim = hidden_dim or xnmt_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or xnmt_global.dropout
    self.weightnoise_std = weightnoise_std or xnmt_global.weight_noise
    self.input_dim = input_dim

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.Wx = dy.parameter(self.p_Wx)
    self.Wh = dy.parameter(self.p_Wh)
    self.b = dy.parameter(self.p_b)
    self.dropout_mask_x = None
    self.dropout_mask_h = None    

  def get_final_states(self):
    return self._final_states

  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def __call__(self, expr_seq):
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    :param expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    :returns: expression sequence
    """
    if isinstance(expr_seq, ExpressionSequence):
      expr_seq = [expr_seq]
    batch_size = expr_seq[0][0].dim()[1]
    seq_len = len(expr_seq[0])
    
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=batch_size)]
    for pos_i in range(seq_len):
      x_t = [expr_seq[j][pos_i] for j in range(len(expr_seq))]
      if isinstance(x_t, dy.Expression):
        x_t = [x_t]
      elif type(x_t) != list:
        x_t = list(x_t)
      if self.dropout_rate > 0.0 and self.train:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates_t = dy.vanilla_lstm_gates_dropout_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std if self.train else 0.0)
      else:
        gates_t = dy.vanilla_lstm_gates_concat(x_t, h[-1], self.Wx, self.Wh, self.b, self.weightnoise_std if self.train else 0.0)
      c_t = dy.vanilla_lstm_c(c[-1], gates_t)
      h_t = dy.vanilla_lstm_h(c_t, gates_t)
      if expr_seq[0].mask is None or np.isclose(np.sum(expr_seq[0].mask.np_arr[:,pos_i:pos_i+1]), 0.0):
        c.append(c_t)
        h.append(h_t)
      else:
        c.append(expr_seq[0].mask.cmult_by_timestep_expr(c_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(c[-1],pos_i,False))
        h.append(expr_seq[0].mask.cmult_by_timestep_expr(h_t,pos_i,True) + expr_seq[0].mask.cmult_by_timestep_expr(h[-1],pos_i,False))
    self._final_states = [FinalTransducerState(h[-1], c[-1])]
    return ExpressionSequence(expr_list=h[1:], mask=expr_seq[0].mask)

class BiLSTMSeqTransducer(SeqTransducer, Serializable):
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than the native CompactVanillaLSTMBuilder due to avoiding concat operations.
  """
  yaml_tag = u'!BiLSTMSeqTransducer'
  
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), layers=1, input_dim=None, hidden_dim=None, dropout=None, weightnoise_std=None):
    register_handler(self)
    self.num_layers = layers
    input_dim = input_dim or xnmt_global.default_layer_dim
    hidden_dim = hidden_dim or xnmt_global.default_layer_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout or xnmt_global.dropout
    self.weightnoise_std = weightnoise_std or xnmt_global.weight_noise
    assert hidden_dim % 2 == 0
    self.forward_layers = [UniLSTMSeqTransducer(xnmt_global=xnmt_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std)]
    self.backward_layers = [UniLSTMSeqTransducer(xnmt_global=xnmt_global, input_dim=input_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std)]
    self.forward_layers += [UniLSTMSeqTransducer(xnmt_global=xnmt_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std) for _ in range(layers-1)]
    self.backward_layers += [UniLSTMSeqTransducer(xnmt_global=xnmt_global, input_dim=hidden_dim, hidden_dim=hidden_dim/2, dropout=dropout, weightnoise_std=weightnoise_std) for _ in range(layers-1)]

  @handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def __call__(self, es):
    mask = es.mask
    # first layer
    forward_es = self.forward_layers[0](es)
    rev_backward_es = self.backward_layers[0](ReversedExpressionSequence(es))

    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i]([forward_es, ReversedExpressionSequence(rev_backward_es)])
      rev_backward_es = ExpressionSequence(self.backward_layers[layer_i]([ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(), mask=mask)
      forward_es = new_forward_es

    self._final_states = [FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].main_expr()]),
                                            dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                            self.backward_layers[layer_i].get_final_states()[0].cell_expr()])) \
                          for layer_i in range(len(self.forward_layers))]
    return ExpressionSequence(expr_list=[dy.concatenate([forward_es[i],rev_backward_es[-i-1]]) for i in range(len(forward_es))], mask=mask)


class CustomLSTMSeqTransducer(SeqTransducer):
  """
  This implements an LSTM builder based on elementary DyNet operations.
  It is more memory-hungry than the compact LSTM, but can be extended more easily.
  It currently does not support dropout or multiple layers and is mostly meant as a
  starting point for LSTM extensions.
  """
  def __init__(self, layers, input_dim, hidden_dim, xnmt_global=Ref(Path("xnmt_global"))):
    if layers!=1: raise RuntimeError("CustomLSTMSeqTransducer supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = xnmt_global.dynet_param_collection.param_col

    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))

  def __call__(self, xs):
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

