import _dynet as dy

class FinalEncoderState(object):
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

class PseudoState(object):
  """
  Emulates a state object for python RNN builders. This allows them to be
  used with minimal changes in code that uses dy.VanillaLSTMBuilder.
  """
  def __init__(self, network, output=None):
    self.network = network
    self._output = output

  def add_input(self, e):
    self._output = self.network.transduce([e])[0]
    return self

  def output(self):
    return self._output

  def h(self):
    raise NotImplementedError("h() is not supported on PseudoStates")

  def s(self):
    raise NotImplementedError("s() is not supported on PseudoStates")
