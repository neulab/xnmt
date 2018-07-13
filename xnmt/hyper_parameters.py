
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable

class Scalar(Serializable):
  """
  Scalar class for hyper parameter that support 1 value serialization.
  This class is actually a base class and does not have any different with simple python float/int.
  
  Args:
    initial: The value being hold by the scalar.
    update: Is the epoch number. 
  """

  yaml_tag = "!Scalar"
  
  @serializable_init
  @register_xnmt_handler
  def __init__(self, initial=0.0, update=0):
    self.value = initial
    self.update = update

  @handle_xnmt_event
  def on_new_epoch(self, *args, **kwargs):
    self.value = self.update_value()
    self.update += 1
    self.save_processed_arg("initial", self.value)
    self.save_processed_arg("update", self.update)
 
  def update_value(self):
    return self.value

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

class DefinedSequence(Scalar):
  """
  Class that represents a fixed defined sequence from config files.
  If update has been made more than the length of the sequence, the last element of the sequence will be returned instead
  
  x = DefinedSequence([0.1, 0.5, 1])
  
  # Epoch 1: 0+x = 0.1
  # Epoch 2: 0+x = 0.5
  # Epoch 3: 0+x = 1
  
  Args:
    sequence: A list of numbers
    initial: The current value or the value.
    update: The epoch number
  """
  
  yaml_tag = "!DefinedSequence"

  @serializable_init
  def __init__(self, sequence=None, initial=0.0, update=0):
    super().__init__(initial, update)
    self.sequence = sequence

  def update_value(self):
    return self.sequence[min(len(self.sequence)-1, self.update)]

