from xnmt.persistence import serializable_init, Serializable
import xnmt.events as events

class ScalingParam(Serializable):
  ''' initial * scaler(epoch-1) '''
  yaml_tag = "!ScalingParam"

  @serializable_init
  def __init__(self, initial=0.0, scaler=None):
    self.__value = initial
    self.scaler = scaler

  def value(self):
    value = self.__value
    if self.scaler is not None:
      value *= self.scaler.value()
    return value

  def __repr__(self):
    return str(self.value())

class Scalar(Serializable):
  yaml_tag = "!Scalar"

  @serializable_init
  def __init__(self, initial=0.0):
    self.__value = initial

  def value(self):
    return self.__value

  def __repr__(self):
    return str(self.value())

class GeometricSequence(Serializable):
  ''' initial^(epoch) '''
  yaml_tag = '!GeometricSequence'

  # Do not set warmup_counter manually.
  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, initial=0.1, warmup=0, ratio=1, min_value=0.0, max_value=1.0):
    self.__value = initial
    self.warmup = warmup
    self.ratio = ratio
    self.min_value = min_value
    self.max_value = max_value
    self.epoch_num = 0

  def value(self):
    if self.epoch_num >= self.warmup:
      return self.__value
    else:
      return 0.0

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    self.epoch_num = training_task.training_state.epoch_num
    if self.epoch_num > self.warmup:
      value = self.__value * self.ratio
      value = max(self.min_value, value)
      value = min(self.max_value, value)
      self.__value = value

  def __repr__(self):
    return repr(self.value())

class DefinedSequence(Serializable):
  yaml_tag = '!DefinedSequence'
  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, sequence=None):
    assert sequence is not None
    assert type(sequence) == list, "DefinedSequence need to have a list type"
    assert len(sequence) > 0, "Please input non empty list for FixedSequence"
    self.sequence = sequence
    self.epoch_num = 0

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, *args, **kwargs):
    self.epoch_num = training_task.training_state.epoch_num

  def __repr__(self):
    return repr(self.value())

  def value(self):
    if self.epoch_num >= len(self.sequence):
      return self.sequence[-1]
    else:
      return self.sequence[self.epoch_num]

