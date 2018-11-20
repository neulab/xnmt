from typing import List
import numbers

import dynet as dy

from xnmt.modelparts import transforms
from xnmt.persistence import serializable_init, Serializable
from xnmt import expression_seqs

class FinalTransducerState(object):
  """
  Represents the final encoder state; Currently handles a main (hidden) state and a cell
  state. If cell state is not provided, it is created as tanh^{-1}(hidden state).
  Could in the future be extended to handle dimensions other than h and c.

  Args:
    main_expr: expression for hidden state
    cell_expr: expression for cell state, if exists
  """
  def __init__(self, main_expr: dy.Expression, cell_expr: dy.Expression=None) -> None:
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
  A class that transforms one sequence of vectors into another, using :class:`expression_seqs.ExpressionSequence` objects as inputs and outputs.
  """

  def transduce(self, seq: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    """
    Parameters should be :class:`expression_seqs.ExpressionSequence` objects wherever appropriate

    Args:
      seq: An expression sequence representing the input to the transduction

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
  accept exactly one argument (an :class:`expression_seqs.ExpressionSequence`) in their transduce method.
  
  Args:
    input_dim: input dimension (not required)
    modules: list of SeqTransducer modules
  """

  yaml_tag = '!ModularSeqTransducer'

  @serializable_init
  def __init__(self, input_dim: numbers.Integral, modules: List[SeqTransducer]):
    self.modules = modules

  def shared_params(self):
    return [{".input_dim", ".modules.0.input_dim"}]

  def transduce(self, seq: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
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
  def __init__(self) -> None:
    pass

  def transduce(self, seq: 'expression_seqs.ExpressionSequence') -> 'expression_seqs.ExpressionSequence':
    return seq


class TransformSeqTransducer(SeqTransducer, Serializable):
  """
  A sequence transducer that applies a given transformation to the sequence's tensor representation

  Args:
      transform: the Transform to apply to the sequence
      downsample_by: if > 1, downsample the sequence via appropriate reshapes.
                     The transform must accept a respectively larger hidden dimension.
  """
  yaml_tag = '!TransformSeqTransducer'

  @serializable_init
  def __init__(self, transform: transforms.Transform, downsample_by: numbers.Integral = 1) -> None:
    self.transform = transform
    if downsample_by < 1: raise ValueError(f"downsample_by must be >=1, was {downsample_by}")
    self.downsample_by = downsample_by

  def get_final_states(self) -> List[FinalTransducerState]:
    return self._final_states

  def transduce(self, src: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    src_tensor = src.as_tensor()
    out_mask = src.mask
    if self.downsample_by > 1:
      assert len(src_tensor.dim()[0])==2, \
        f"Downsampling only supported for tensors of order to. Found dims {src_tensor.dim()}"
      (hidden_dim, seq_len), batch_size = src_tensor.dim()
      if seq_len % self.downsample_by != 0:
        raise ValueError(
          "For downsampling, sequence lengths must be multiples of the total reduce factor. "
          "Configure batcher accordingly.")
      src_tensor = dy.reshape(src_tensor,
                              (hidden_dim*self.downsample_by, seq_len//self.downsample_by),
                              batch_size=batch_size)
      if out_mask:
        out_mask = out_mask.lin_subsampled(reduce_factor=self.downsample_by)
    output = self.transform.transform(src_tensor)
    if self.downsample_by==1:
      if len(output.dim())!=src_tensor.dim(): # can happen with seq length 1
        output = dy.reshape(output, src_tensor.dim()[0], batch_size=src_tensor.dim()[1])
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output, mask=out_mask)
    self._final_states = [FinalTransducerState(output_seq[-1])]
    return output_seq
