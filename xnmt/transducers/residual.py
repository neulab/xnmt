from typing import List
import numbers

import dynet as dy

from xnmt import events, expression_seqs, param_collections
from xnmt.transducers import base as transducers
from xnmt.persistence import Ref, serializable_init, Serializable

class ResidualSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  A sequence transducer that wraps a :class:`xnmt.transducers.base.SeqTransducer` in an additive residual
  connection, and optionally performs some variety of normalization.

  Args:
    child the child transducer to wrap
    layer_norm: whether to perform layer normalization
    dropout: whether to apply residual dropout
  """

  yaml_tag = '!ResidualSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, child: transducers.SeqTransducer, input_dim: numbers.Integral, layer_norm: bool = False,
               dropout=Ref("exp_global.dropout", default=0.0)) -> None:
    self.child = child
    self.dropout = dropout
    self.input_dim = input_dim
    self.layer_norm = layer_norm
    if layer_norm:
      model = param_collections.ParamManager.my_params(self)
      self.ln_g = model.add_parameters(dim=(input_dim,))
      self.ln_b = model.add_parameters(dim=(input_dim,))

  @ events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def transduce(self, seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:

    if self.train and self.dropout > 0.0:
      seq_tensor = dy.dropout(self.child.transduce(seq).as_tensor(), self.dropout) + seq.as_tensor()
    else:
      seq_tensor = self.child.transduce(seq).as_tensor() + seq.as_tensor()
    if self.layer_norm:
      d = seq_tensor.dim()
      seq_tensor = dy.reshape(seq_tensor, (d[0][0],), batch_size=d[0][1]*d[1])
      seq_tensor = dy.layer_norm(seq_tensor, self.ln_g, self.ln_b)
      seq_tensor = dy.reshape(seq_tensor, d[0], batch_size=d[1])
    return expression_seqs.ExpressionSequence(expr_tensor=seq_tensor)

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    # TODO: is this OK to do?
    return self.child.get_final_states()

