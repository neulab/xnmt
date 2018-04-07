import dynet as dy

from xnmt.serialize.serializer import serializable_init, Serializable, Path
from xnmt.expression_sequence import ExpressionSequence

class Transducer(object):
  """
  A transducer takes an input and transduces it into some output.

  Inputs and outputs can be DyNet expressions, :class:`xnmt.expression_sequence.ExpressionSequence`, or anything else.

  The goal is to make XNMT as modular as possible. It will be the responsibility of
  the user to ensure a transducer is being used in the correct context (i.e., supports
  correct input and output data types), therefore the expected data types should be made
  clear via appropriate argument naming and documentation.

  Transducers in general will have at least two methods:
  - __init__(...), should be used to configure the transducer. If possible, configuration
  should be transparent to a user and not require understanding of implementation
  details. If the transducer uses DyNet parameters, these must be initialized here.
  - __call__(...), will perform the actual transduction and return the result
  """
  def __call__(self, *args, **kwargs):
    """
    May take any parameters.
    
    Returns:
      result of transduction
    """
    raise NotImplementedError("subclasses must implement __call__()")

class SeqTransducer(Transducer):
  """
  A special type of :class:`xnmt.transducer.Transducer` that uses :class:`xnmt.expression_sequence.ExpressionSequence` objects as inputs and outputs.
  Whenever dealing with sequences, :class:`xnmt.expression_sequence.ExpressionSequence` are preferred because they are more
  powerful and flexible than plain DyNet expressions.
  """
  def __call__(self, *args, **kwargs):
    """
    Parameters should be :class:`xnmt.expression_sequence.ExpressionSequence` objects wherever appropriate

    Returns:
      result of transduction, an :class:`xnmt.expression_sequence.ExpressionSequence` object
    """
    raise NotImplementedError("subclasses must implement __call__()")

  def get_final_states(self):
    """Returns:
         A list of FinalTransducerState objects corresponding to a fixed-dimension representation of the input, after having invoked __call__()
    """
    return []


class FinalTransducerState(object):
  """
  Represents the final encoder state; Currently handles a main (hidden) state and a cell
  state. If cell state is not provided, it is created as tanh^{-1}(hidden state).
  Could in the future be extended to handle dimensions other than h and c.
  
  Args:
    main_expr (dy.Expression): expression for hidden state
    cell_expr (dy.Expression): expression for cell state, if exists
  """
  def __init__(self, main_expr, cell_expr=None):
    self._main_expr = main_expr
    self._cell_expr = cell_expr
  def main_expr(self): return self._main_expr
  def cell_expr(self):
    """Returns:
         dy.Expression: cell state; if not given, it is inferred as inverse tanh of main expression
    """
    if self._cell_expr is None:
      self._cell_expr = 0.5 * dy.log( dy.cdiv(1.+self._main_expr, 1.-self._main_expr) )
    return self._cell_expr


########################################################

class ModularSeqTransducer(SeqTransducer, Serializable):
  """
  A sequence transducer that stacks several :class:`xnmt.transducer.SeqTransducer` objects, all of which must
  accept exactly one argument (an :class:`xnmt.expression_sequence.ExpressionSequence`) in their __call__ method.
  
  Args:
    input_dim (int): input dimension (not required)
    modules (list of :class:`xnmt.transducer.SeqTransducer`): list of SeqTransducer modules
  """

  yaml_tag = '!ModularSeqTransducer'

  @serializable_init
  def __init__(self, input_dim, modules):
    self.modules = modules

  def shared_params(self):
    return [set([Path(".input_dim"), Path(".modules.0.input_dim")])]

  def __call__(self, es):
    for module in self.modules:
      es = module(es)
    return es

  def get_final_states(self):
    final_states = []
    for mod in self.modules:
      final_states += mod.get_final_states()
    return final_states


class IdentitySeqTransducer(Transducer, Serializable):
  """
  A transducer that simply returns the input.
  """

  yaml_tag = '!IdentitySeqTransducer'

  @serializable_init
  def __call__(self, output):
    if not isinstance(output, ExpressionSequence):
      output = ExpressionSequence(expr_list=output)
    return output

