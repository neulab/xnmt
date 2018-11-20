from typing import Optional, Dict, List, Tuple
import collections

import dynet as dy

class FactoredLossExpr(object):
  
  """
  Loss consisting of (possibly batched) DyNet expressions, with one expression per loss factor.

  Used to represent losses within a training step.

  Args:
    init_loss: initial loss values
  """

  def __init__(self, init_loss: Optional[Dict[str, dy.Expression]] = None) -> None:
    self.expr_factors = collections.defaultdict(lambda: dy.scalarInput(0))
    if init_loss is not None:
      for key, val in init_loss.items():
        self.expr_factors[key] = val

  def add_loss(self, loss_name: str, loss: Optional[dy.Expression]) -> None:
    if loss:
      self.expr_factors[loss_name] += loss

  def add_factored_loss_expr(self, factored_loss_expr: Optional['FactoredLossExpr']) -> None:
    if factored_loss_expr:
      for loss_name, loss in factored_loss_expr.expr_factors.items():
        self.expr_factors[loss_name] += loss

  def compute(self, comb_method: str = "sum") -> dy.Expression:
    """
    Compute loss as DyNet expression by summing over factors and batch elements.

    Args:
      comb_method: method for combining loss across batch elements ('sum' or 'avg').

    Returns:
      Scalar DyNet expression.
    """
    return self._combine_batches(dy.esum(list(self.expr_factors.values())), comb_method)

  def value(self) -> List[float]:
    """
    Get list of per-batch-element loss values, summed over factors.

    Returns:
      List of same length as batch-size.
    """
    return dy.esum(list(self.expr_factors.values())).value()

  def __getitem__(self, loss_name: str) -> dy.Expression:
    return self.expr_factors[loss_name]

  def get_factored_loss_val(self, comb_method: str = "sum") -> 'FactoredLossVal':
    """
    Create factored loss values by calling ``.value()`` for each DyNet loss expression and applying batch combination.

    Args:
      comb_method: method for combining loss across batch elements ('sum' or 'avg').

    Returns:
      Factored loss values.
    """
    return FactoredLossVal({k: self._combine_batches(v, comb_method).value() for k, v in self.expr_factors.items()})

  def _combine_batches(self, batched_expr, comb_method: str = "sum"):
    if comb_method == "sum":
      return dy.sum_batches(batched_expr)
    elif comb_method == "avg":
      return dy.sum_batches(batched_expr) * (1.0 / batched_expr.dim()[1])
    else:
      raise ValueError(f"Unknown batch combination method '{comb_method}', expected 'sum' or 'avg'.'")

  def get_nobackprop_loss(self) -> Dict[str, dy.Expression]:
    """
    Get dictionary of named non-backpropagating loss expressions

    Returns:
      Loss expressions
    """
    return {k: dy.nobackprop(v) for k, v in self.expr_factors.items()}

  def __len__(self):
    return len(self.expr_factors)

  def __mul__(self, scalar):
    return FactoredLossExpr({key: scalar*value for key, value in self.expr_factors.items()})

class FactoredLossVal(object):
  
  """
  Loss consisting of (unbatched) float values, with one value per loss factor.

  Used to represent losses accumulated across several training steps.
  """

  def __init__(self, loss_dict = None) -> None:
    if loss_dict is None:
      loss_dict = {}
    self._loss_dict = loss_dict

  def __iadd__(self, other: 'FactoredLossVal') -> 'FactoredLossVal':
    """
    Implements += operator, adding up factors individually.

    Args:
      other: other factored float loss

    Returns:
      self
    """
    for name, value in other._loss_dict.items():
      if name in self._loss_dict:
        self._loss_dict[name] += value
      else:
        self._loss_dict[name] = value
    return self

  def sum_factors(self) -> float:
    """
    Return the sum of all loss factors.

    Returns:
      A float value.
    """
    return sum([x for x in self._loss_dict.values()])

  def items(self) -> List[Tuple[str, float]]:
    """
    Get name/value tuples for loss factors.

    Returns:
      Name/value tuples.
    """
    return self._loss_dict.items()

  def __len__(self):
    return len(self._loss_dict)

  def clear(self) -> None:
    """
    Clears all loss factors.
    """
    self._loss_dict.clear()

