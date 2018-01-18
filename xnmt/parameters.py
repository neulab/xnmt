
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.reports import Reportable
from xnmt.serializer import Serializable

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

  def __init__(self, value=0.0):
    self.__value = value

  def value(self):
    return self.__value

class GeometricSequence(Serializable):
  ''' initial^(epoch) '''
  yaml_tag = u'!GeometricSequence'

  # Do not set warmup_counter manually.
  def __init__(self, initial=0.1, warmup=0, ratio=1, min_value=0.0, max_value=1.0, warmup_counter=0):
    register_handler(self)
    self.__value = initial
    self.warmup = warmup
    self.warmup_counter = warmup_counter
    self.ratio = ratio
    self.min_value = min_value
    self.max_value = max_value

  def value(self):
    if self.warmup_counter >= self.warmup:
      return self.__value
    else:
      return 0.0

  @handle_xnmt_event
  def on_next_epoch(self):
    self.warmup_counter += 1
    if self.warmup_counter >= self.warmup:
      value = self.__value * self.ratio
      value = max(self.min_value, value)
      value = min(self.max_value, value)
      self.__value = value

  def __repr__(self):
    return str(self.__value)

