from typing import List
import numbers

import xnmt
import xnmt.tensor_tools as tt
from xnmt import events, expression_seqs
from xnmt.transducers import base as transducers
from xnmt.persistence import Ref, serializable_init, Serializable

class ResidualSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  A sequence transducer that wraps a :class:`xnmt.transducers.base.SeqTransducer` in an additive residual
  connection, and optionally performs some variety of normalization.

  Args:
    child: the child transducer to wrap
    layer_norm: whether to perform layer normalization
    dropout: whether to apply residual dropout
  """

  yaml_tag = '!ResidualSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, child: transducers.SeqTransducer, input_dim: numbers.Integral, layer_norm: bool = False,
               dropout=Ref("exp_global.dropout", default=0.0), layer_norm_component: xnmt.norms.LayerNorm = None) -> None:
    self.child = child
    self.dropout = dropout
    self.input_dim = input_dim
    self.layer_norm = layer_norm
    if layer_norm:
      self.layer_norm_component = self.add_serializable_component("layer_norm_component",
                                                                  layer_norm_component,
                                                                  lambda: xnmt.norms.LayerNorm(hidden_dim=input_dim))

  @ events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def transduce(self, seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:

    if self.train and self.dropout > 0.0:
      seq_tensor = tt.dropout(self.child.transduce(seq).as_tensor(), self.dropout) + seq.as_tensor()
    else:
      seq_tensor = self.child.transduce(seq).as_tensor() + seq.as_tensor()
    if self.layer_norm:
      batch_size = tt.batch_size(seq_tensor)
      merged_seq_tensor = tt.merge_time_batch_dims(seq_tensor)
      transformed_seq_tensor = self.layer_norm_component.transform(merged_seq_tensor)
      seq_tensor = tt.unmerge_time_batch_dims(transformed_seq_tensor, batch_size)
    return expression_seqs.ExpressionSequence(expr_tensor=seq_tensor)

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    # TODO: is this OK to do?
    return self.child.get_final_states()

