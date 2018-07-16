from typing import List
import dynet as dy

from xnmt.persistence import serializable_init, Serializable
from xnmt.expression_sequence import ExpressionSequence

class FinalTransducerState(object):
  """
  Represents the final encoder state; Currently handles a main (hidden) state and a cell
  state. If cell state is not provided, it is created as tanh^{-1}(hidden state).
  Could in the future be extended to handle dimensions other than h and c.

  Args:
    main_expr: expression for hidden state
    cell_expr: expression for cell state, if exists
  """
  def __init__(self, main_expr: dy.Expression, cell_expr: dy.Expression=None):
    self._main_expr = main_expr
    self._cell_expr = cell_expr

  def main_expr(self) -> dy.Expression:
    return self._main_expr

  def cell_expr(self) -> dy.Expression:
    """Returns:
         dy.Expression: cell state; if not given, it is inferred as inverse tanh of main expression
    """
    if self._cell_expr is None:
      # TODO: This taking of the tanh inverse is disabled, because it can cause NaNs
      #       Instead just copy
      # self._cell_expr = 0.5 * dy.log( dy.cdiv(1.+self._main_expr, 1.-self._main_expr) )
      self._cell_expr = self._main_expr
    return self._cell_expr

class SeqTransducer(object):
  """
  A class that transforms one sequence of vectors into another, using :class:`xnmt.expression_sequence.ExpressionSequence` objects as inputs and outputs.
  """

  def transduce(self, seq: ExpressionSequence) -> ExpressionSequence:
    """
    Parameters should be :class:`xnmt.expression_sequence.ExpressionSequence` objects wherever appropriate

    Args:
      seq: An ExpressionSequence representing the input to the transduction

    Returns:
      result of transduction, an expression sequence
    """
    raise NotImplementedError("SeqTransducer.transduce() must be implemented by SeqTransducer sub-classes")

  def get_final_states(self) -> List[FinalTransducerState]:
    """Returns:
         A list of FinalTransducerState objects corresponding to a fixed-dimension representation of the input, after having invoked transduce()
    """
    raise NotImplementedError("SeqTransducer.get_final_states() must be implemented by SeqTransducer sub-classes")


########################################################

class ModularSeqTransducer(SeqTransducer, Serializable):
  """
  A sequence transducer that stacks several :class:`xnmt.transducer.SeqTransducer` objects, all of which must
  accept exactly one argument (an :class:`xnmt.expression_sequence.ExpressionSequence`) in their transduce method.
  
  Args:
    input_dim (int): input dimension (not required)
    modules (list of :class:`xnmt.transducer.SeqTransducer`): list of SeqTransducer modules
  """

  yaml_tag = '!ModularSeqTransducer'

  @serializable_init
  def __init__(self, input_dim: int, modules: List[SeqTransducer]):
    self.modules = modules

  def shared_params(self):
    return [{".input_dim", ".modules.0.input_dim"}]

  def transduce(self, seq: ExpressionSequence) -> ExpressionSequence:
    for module in self.modules:
      seq = module.transduce(seq)
    return seq

  def get_final_states(self) -> List[FinalTransducerState]:
    final_states = []
    for mod in self.modules:
      final_states += mod.get_final_states()
    return final_states


class IdentitySeqTransducer(SeqTransducer, Serializable):
  """
  A transducer that simply returns the input.
  """

  yaml_tag = '!IdentitySeqTransducer'

  @serializable_init
  def __init__(self):
    pass

  def transduce(self, seq: ExpressionSequence) -> ExpressionSequence:
    return seq

