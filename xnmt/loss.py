import dynet as dy
import collections

class LossBuilder(object):

  # TODO: document me

  def __init__(self, init_loss=None):
    self.loss_values = collections.defaultdict(lambda: dy.scalarInput(0))
    self.modified = True
    if init_loss != None:
      for key, val in init_loss.items():
        self.loss_values[key] = val

  def add_loss(self, loss_name, loss_expr):
    if loss_expr is None:
      return
    self.modified = True
    if type(loss_expr) == LossBuilder:
      for loss_name, loss in loss_expr.loss_values.items():
        self.loss_values[loss_name] += loss
    else:
      self.loss_values[loss_name] += loss_expr

  def delete_loss(self, loss_name):
    self.modified = True
    loss = self.loss_values[loss_name]
    del self.loss_values[loss_name]
    return loss

  def compute(self):
    return dy.sum_batches(self.sum())

  def set_loss(self, loss_name, loss_value):
    self.modified = True
    self.loss_values[loss_name] = loss_value

  def get_loss(self, loss_name):
    return self.loss_values.get(loss_name, None)

  def sum(self, batch_sum=True):
    if self.modified:
      self.sum_value = dy.esum(list(self.loss_values.values()))
      self.modified = False
    return self.sum_value

  def value(self):
    return self.sum().value()

  def __getitem__(self, index):
    return self.loss_values[index]

  def get_loss_stats(self):
    return LossScalarBuilder({k: dy.sum_batches(v).value() for k, v in self.loss_values.items()})

  def __len__(self):
    return len(self.loss_values)

  def __contains__(self, item):
    return item in self.loss_values

  def __repr__(self):
    loss_str = ", ".join(["%s %f" % (loss_name, dy.sum_batches(loss_value).value()) for loss_name, loss_value in self.loss_values.items()])
    return "{Loss Builder: %s}" % (loss_str)

class LossScalarBuilder(object):

  # TODO: document me

  def __init__(self, loss_stats=None):
    if loss_stats == None:
      loss_stats = {}
    self.__loss_stats = loss_stats

  def __iadd__(self, other):
    for name, value in other.__loss_stats.items():
      if name in self.__loss_stats:
        self.__loss_stats[name] += value
      else:
        self.__loss_stats[name] = value
    return self

  def sum(self):
    return sum([x for x in self.__loss_stats.values()])

  def items(self):
    return self.__loss_stats.items()

  def __len__(self):
    return len(self.__loss_stats)

  def zero(self):
    self.__loss_stats.clear()

