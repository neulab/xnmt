import dynet as dy
from typing import List

from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.expression_sequence import ExpressionSequence, ReversedExpressionSequence
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducer import SeqTransducer, FinalTransducerState
from xnmt.param_collection import ParamManager

class ResidualSeqTransducer(SeqTransducer, Serializable):
  """
  A sequence transducer that wraps a :class:`xnmt.transducer.SeqTransducer` in an additive residual
  connection, and optionally performs some variety of normalization.

  Args:
    child (:class:`xnmt.transducer.SeqTransducer`): the child transducer to wrap
    layer_norm (bool): whether to perform layer normalization
  """

  yaml_tag = '!ResidualSeqTransducer'

  @serializable_init
  def __init__(self, child: SeqTransducer, input_dim: int, layer_norm: bool = False):
    self.child = child
    self.input_dim = input_dim
    self.layer_norm = layer_norm
    if layer_norm:
      model = ParamManager.my_params(self)
      self.ln_g = model.add_parameters(dim=(input_dim,))
      self.ln_b = model.add_parameters(dim=(input_dim,))

  def transduce(self, seq: ExpressionSequence) -> ExpressionSequence:
    seq_tensor = self.child.transduce(seq).as_tensor() + seq.as_tensor()
    if self.layer_norm:
      d = seq_tensor.dim()
      seq_tensor = dy.reshape(seq_tensor, (d[0][0],), batch_size=d[0][1]*d[1])
      seq_tensor = dy.layer_norm(seq_tensor, self.ln_g, self.ln_b)
      seq_tensor = dy.reshape(seq_tensor, d[0], batch_size=d[1])
    return ExpressionSequence(expr_tensor=seq_tensor)

  def get_final_states(self) -> List[FinalTransducerState]:
    # TODO: is this OK to do?
    return self.child.get_final_states()

