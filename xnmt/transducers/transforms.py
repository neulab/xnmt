from typing import List

from xnmt.expression_seqs import ExpressionSequence
from xnmt.persistence import Serializable, serializable_init
from xnmt.transducers.base import SeqTransducer, FinalTransducerState
from xnmt.transforms import Transform


class TransformSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!TransformSeqTransducer'

  @serializable_init
  def __init__(self, transform: Transform):
    """
    Args:
      transform: the Transform to apply to the sequence
    """
    self.transform = transform

  def get_final_states(self) -> List[FinalTransducerState]:
    return self._final_states

  def transduce(self, src: ExpressionSequence) -> ExpressionSequence:
    output = self.transform(src.as_tensor())
    output_seq = ExpressionSequence(expr_tensor=output)
    self._final_states = [FinalTransducerState(output_seq[-1])]
    return output_seq