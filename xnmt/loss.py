from __future__ import print_function

import sys
import dynet as dy
import collections

class LossBuilder(object):
  def __init__(self):
    self.loss_values = collections.defaultdict(lambda: dy.scalarInput(0))

  def add_loss(self, loss_name, loss_expr):
    if loss_expr is None:
      return
    if type(loss_expr) == LossBuilder:
      for loss_name, loss in loss_expr.loss_values.items():
        self.loss_values[loss_name] += loss
    else:
      self.loss_values[loss_name] += loss_expr

  def compute(self):
    try:
      return dy.sum_batches(dy.esum(list(self.loss_values.values())))
    finally:
      self.loss_values.clear()

  def __getitem__(self, index):
    return self.loss_values[index]

  def get_loss_stats(self):
    return LossScalarBuilder({k: v.value() for k, v in self.loss_values.items()})

  def __len__(self):
    return len(self.loss_values)

  def __repr__(self):
    self.compute()
    loss_str = ", ".join(["%s %f" % (loss_name, loss_value.value()) for loss_name, loss_value in self.loss_values.items()])
    return "{Loss Builder: %s}" % (loss_str)

class LossScalarBuilder(object):
  def __init__(self, loss_stats={}):
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

