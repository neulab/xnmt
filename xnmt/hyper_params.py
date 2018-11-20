import typing
import numbers

from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable

class Scalar(Serializable):
  """
  Scalar class for hyper parameter that support 1 value serialization.
  This class is actually a base class and does not have any different with simple python float/int.
  
  Args:
    initial: The value being hold by the scalar.
    times_updated: Is the epoch number.
  """

  yaml_tag = "!Scalar"
  
  @serializable_init
  @register_xnmt_handler
  def __init__(self, initial: numbers.Real = 0.0, times_updated: numbers.Integral = 0) -> None:
    self.initial = initial
    self.times_updated = times_updated
    self.value = self.get_curr_value()

  @handle_xnmt_event
  def on_new_epoch(self, *args, **kwargs):
    self.value = self.get_curr_value()
    self.times_updated += 1
    self.save_processed_arg("times_updated", self.times_updated)
 
  def get_curr_value(self) -> numbers.Real:
    return self.initial

  def __repr__(self):
    return f"{self.__class__.__name__}[curr={self.get_curr_value()}]"

  # Operators
  def __lt__(a, b): return a.value < b
  def __le__(a, b): return a.value <= b
  def __eq__(a, b): return a.value == b
  def __ne__(a, b): return a.value != b
  def __ge__(a, b): return a.value >= b
  def __gt__(a, b): return a.value > b
  def __add__(a, b): return a.value + b
  def __sub__(a, b): return a.value - b
  def __mul__(a, b): return a.value * b
  def __neg__(a): return -a.value
  def __pos__(a): return +a.value
  def __pow__(a, b): return a.value ** b
  def __truediv__(a, b): return a.value / b
  def __floordiv__(a, b): return a.value // b

class DefinedSequence(Scalar, Serializable):
  """
  Class that represents a fixed defined sequence from config files.
  If update has been made more than the length of the sequence, the last element of the sequence will be returned instead
  
  x = DefinedSequence([0.1, 0.5, 1])
  
  # Epoch 1: 0.1
  # Epoch 2: 0.5
  # Epoch 3: 1
  # Epoch 4: 1
  # ...

  Args:
    sequence: A list of numbers
    times_updated: The epoch number
  """
  
  yaml_tag = "!DefinedSequence"

  @serializable_init
  def __init__(self, sequence: typing.Sequence[numbers.Real], times_updated: numbers.Integral = 0) -> None:
    self.sequence = sequence
    if len(sequence)==0: raise ValueError("DefinedSequence initialized with empty sequence")
    super().__init__(times_updated=times_updated)

  def get_curr_value(self) -> numbers.Real:
    return self.sequence[min(len(self.sequence) - 1, self.times_updated)]

numbers.Real.register(Scalar)
