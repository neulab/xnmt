
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.reports import Reportable
from xnmt.serialize.serializable import Serializable

class ScalingParam(Serializable):
  ''' initial * scaler(epoch-1) '''
  yaml_tag = u"!ScalingParam"

  def __init__(self, initial=0.0, scaler=None):
    self.__value = initial
    self.scaler = scaler

  def value(self):
    value = self.__value
    if self.scaler is not None:
      value *= self.scaler.value()
    return value

class Scalar(Serializable):
  yaml_tag = u"!Scalar"

  def __init__(self, initial=0.0):
    self.__value = initial

  def value(self):
    return self.__value

class GeometricSequence(Serializable):
  ''' initial^(epoch) '''
  yaml_tag = u'!GeometricSequence'

  # Do not set warmup_counter manually.
  def __init__(self, initial=0.1, warmup=0, ratio=1, min=0.0, max=1.0):
    register_handler(self)
    self.__value = initial
    self.warmup = warmup
    self.ratio = ratio
    self.min_value = min
    self.max_value = max
    self.epoch_num = 0

  def value(self):
    if hasattr(self, "epoch_num") and self.epoch_num >= self.warmup:
      return self.__value
    else:
      return 0.0

  @handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    self.epoch_num = training_task.training_state.epoch_num
    if self.epoch_num > self.warmup:
      value = self.__value * self.ratio
      value = max(self.min_value, value)
      value = min(self.max_value, value)
      self.__value = value

class DefinedSequence(Serializable):
  yaml_tag = u'!DefinedSequence'
  def __init__(self, sequence=None):
    assert sequence is not None
    assert type(sequence) == list, "DefinedSequence need to have a list type"
    assert len(sequence) > 0, "Please input non empty list for FixedSequence"
    register_handler(self)
    self.sequence = sequence
    self.epoch_num = 0

  @handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    self.epoch_num = training_task.training_state.epoch_num

  def value(self):
    if self.epoch_num >= len(self.sequence):
      return self.sequence[-1]
    else:
      return self.sequence[self.epoch_num-1]

def multiply_weight(value, weight):
  weight = weight if weight is not None else 1
  if weight is not None and hasattr(weight, "value"):
    weight = weight.value()
  if weight < 1e-8:
    return None
  else:
    return value * weight
