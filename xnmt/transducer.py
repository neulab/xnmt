from __future__ import division, generators

import dynet as dy

from xnmt.serializer import Serializable

class Transducer(object):
  """
  A transducer takes an input and transduces it into some output.
  
  Inputs and outputs can be DyNet expressions, ExpressionSequences, or anything else.
  
  The goal is to make XNMT as modular as possible. It will be the responsibility of
  the user to ensure a transducer is being used in the correct context (i.e., supports
  correct input and output data types), therefore the expected data types should be made
  clear via appropriate argument naming and documentation.
  
  Transducers in general will have at least two methods:
  - __init__(...), should be used to configure the transducer. If possible, configuration
    should be transparent to a user and not require understanding of implementation
    details. If the transducer uses DyNet parameters, these must be initialized here.
    If appropriate, yaml_context argument should be used to access global configuration
    and DyNet parameters
  - __call__(...), will perform the actual transduction and return the result 
  """
  def __call__(self, *args, **kwargs):
    """
    May take any parameters.
    :returns: result of transduction
    """
    raise NotImplementedError("subclasses must implement __call__()")

class SeqTransducer(Transducer):
  """
  A special type of transducer that uses ExpressionSequence objects as inputs and outputs.
  Whenever dealing with sequences, ExpressionSequences are preferred because they are more
  powerful and flexible than plain DyNet expressions.
  """
  def __call__(self, *args, **kwargs):
    """
    Parameters should be ExpressionSequence objects wherever appropriate
    :returns: result of transduction, an ExpressionSequence object
    """
    raise NotImplementedError("subclasses must implement __call__()")
  
  def get_final_states(self):
    """
    :returns: A list of FinalTransducerState objects corresponding to a fixed-dimension
              representation of the input, after having invoked __call__()
    """
    return []


class FinalTransducerState(object):
  """
  Represents the final encoder state; Currently handles a main (hidden) state and a cell
  state. If cell state is not provided, it is created as tanh^{-1}(hidden state).
  Could in the future be extended to handle dimensions other than h and c.
  """
  def __init__(self, main_expr, cell_expr=None):
    """
    :param main_expr: expression for hidden state
    :param cell_expr: expression for cell state, if exists
    """
    self._main_expr = main_expr
    self._cell_expr = cell_expr
  def main_expr(self): return self._main_expr
  def cell_expr(self):
    """
    returns: cell state; if not given, it is inferred as inverse tanh of main expression
    """
    if self._cell_expr is None:
      self._cell_expr = 0.5 * dy.log( dy.cdiv(1.+self._main_expr, 1.-self._main_expr) )
    return self._cell_expr


########################################################

class ModularSeqTransducer(SeqTransducer, Serializable):
  """
  A sequence transducer that stacks several sequence transducers, all of which must
  accept exactly one argument (an expression sequence) in their __call__ method.
  """

  yaml_tag = u'!ModularSeqTransducer'

  def __init__(self, input_dim, modules):
    self.modules = modules
    
  def shared_params(self):
    return [set(["input_dim", "modules.0.input_dim"])]

  def __call__(self, es):
    for module in self.modules:
      es = module(es)
    return es

  def get_final_states(self):
    final_states = []
    for mod in self.modules:
      final_states += mod.get_final_states()
    return final_states


class IdentityTransducer(Transducer, Serializable):
  """
  A transducer that simply returns the input.
  """
  
  yaml_tag = u'!IdentityTransducer'

  def __call__(self, x):
    return x

