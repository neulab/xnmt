from typing import Any, List, Sequence, Union
import numbers

import dynet as dy

from xnmt.transducers import recurrent
from xnmt import expression_seqs, events
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.transducers import base as transducers


class PyramidalLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  Builder for pyramidal RNNs that delegates to ``UniLSTMSeqTransducer`` objects and wires them together.
  See https://arxiv.org/abs/1508.01211

  Every layer (except the first) reduces sequence length by the specified factor.

  Args:
    layers: number of layers
    input_dim: input dimension
    hidden_dim: hidden dimension
    downsampling_method: how to perform downsampling (concat|skip)
    reduce_factor: integer, or list of ints (different skip for each layer)
    dropout: dropout probability; if None, use exp_global.dropout
    builder_layers: set automatically
  """
  yaml_tag = '!PyramidalLSTMSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               downsampling_method: str = "concat",
               reduce_factor: Union[numbers.Integral, Sequence[numbers.Integral]] = 2,
               dropout: float = Ref("exp_global.dropout", default=0.0),
               builder_layers: Any = None):
    self.dropout = dropout
    assert layers > 0
    assert hidden_dim % 2 == 0
    assert type(reduce_factor)==int or (type(reduce_factor)==list and len(reduce_factor)==layers-1)
    assert downsampling_method in ["concat", "skip"]

    self.downsampling_method = downsampling_method
    self.reduce_factor = reduce_factor
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.builder_layers = self.add_serializable_component("builder_layers", builder_layers,
                                                          lambda: self.make_builder_layers(input_dim, hidden_dim,
                                                                                           layers, dropout,
                                                                                           downsampling_method,
                                                                                           reduce_factor))

  def make_builder_layers(self, input_dim, hidden_dim, layers, dropout, downsampling_method, reduce_factor):
    builder_layers = []
    f = recurrent.UniLSTMSeqTransducer(input_dim=input_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
    b = recurrent.UniLSTMSeqTransducer(input_dim=input_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
    builder_layers.append([f, b])
    for _ in range(layers - 1):
      layer_input_dim = hidden_dim if downsampling_method=="skip" else hidden_dim*reduce_factor
      f = recurrent.UniLSTMSeqTransducer(input_dim=layer_input_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
      b = recurrent.UniLSTMSeqTransducer(input_dim=layer_input_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
      builder_layers.append([f, b])
    return builder_layers

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def _reduce_factor_for_layer(self, layer_i):
    if layer_i >= len(self.builder_layers)-1:
      return 1
    elif type(self.reduce_factor)==int:
      return self.reduce_factor
    else:
      return self.reduce_factor[layer_i]

  def transduce(self, es: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs,
    and concatenating.

    Args:
      es: an ExpressionSequence
    """
    es_list = [es]

    for layer_i, (fb, bb) in enumerate(self.builder_layers):
      reduce_factor = self._reduce_factor_for_layer(layer_i)

      if es_list[0].mask is None: mask_out = None
      else: mask_out = es_list[0].mask.lin_subsampled(reduce_factor)

      if self.downsampling_method=="concat" and len(es_list[0]) % reduce_factor != 0:
        raise ValueError(f"For 'concat' subsampling, sequence lengths must be multiples of the total reduce factor, "
                         f"but got sequence length={len(es_list[0])} for reduce_factor={reduce_factor}. "
                         f"Set Batcher's pad_src_to_multiple argument accordingly.")
      fs = fb.transduce(es_list)
      bs = bb.transduce([expression_seqs.ReversedExpressionSequence(es_item) for es_item in es_list])
      if layer_i < len(self.builder_layers) - 1:
        if self.downsampling_method=="skip":
          es_list = [expression_seqs.ExpressionSequence(expr_list=fs[::reduce_factor], mask=mask_out),
                     expression_seqs.ExpressionSequence(expr_list=bs[::reduce_factor][::-1], mask=mask_out)]
        elif self.downsampling_method=="concat":
          es_len = len(es_list[0])
          es_list_fwd = []
          es_list_bwd = []
          for i in range(0, es_len, reduce_factor):
            for j in range(reduce_factor):
              if i==0:
                es_list_fwd.append([])
                es_list_bwd.append([])
              es_list_fwd[j].append(fs[i+j])
              es_list_bwd[j].append(bs[len(es_list[0])-reduce_factor+j-i])
          es_list = [expression_seqs.ExpressionSequence(expr_list=es_list_fwd[j], mask=mask_out) for j in range(reduce_factor)] + \
                    [expression_seqs.ExpressionSequence(expr_list=es_list_bwd[j], mask=mask_out) for j in range(reduce_factor)]
        else:
          raise RuntimeError(f"unknown downsampling_method {self.downsampling_method}")
      else:
        # concat final outputs
        ret_es = expression_seqs.ExpressionSequence(
          expr_list=[dy.concatenate([f, b]) for f, b in zip(fs, expression_seqs.ReversedExpressionSequence(bs))], mask=mask_out)

    self._final_states = [transducers.FinalTransducerState(dy.concatenate([fb.get_final_states()[0].main_expr(),
                                                                           bb.get_final_states()[0].main_expr()]),
                                                           dy.concatenate([fb.get_final_states()[0].cell_expr(),
                                                                           bb.get_final_states()[0].cell_expr()])) \
                          for (fb, bb) in self.builder_layers]
    return ret_es
