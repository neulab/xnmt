from __future__ import print_function

import sys
import dynet as dy
import collections

class LossBuilder(object):
  def __init__(self):
    self.loss_nodes  = []
    self.loss_values = collections.defaultdict(float)

  def add_loss(self, loss_name, loss_expr):
    if loss_expr.dim()[1] > 1:
      loss_expr = dy.sum_batches(loss_expr)
    self.loss_nodes.append((loss_name, loss_expr))

  def compute(self):
    ''' Compute all the losses and delete the computational graph reference.
    '''
    total_loss = dy.esum([x[1] for x in self.loss_nodes])
    for loss_name, loss_expr in self.loss_nodes:
      self.loss_values[loss_name] += loss_expr.value()
    self.loss_nodes = []
    return total_loss

  def __getitem__(self, index):
    return self.loss_nodes[index][1]

  def __iter__(self):
    return iter(self.loss_values.items())

  def sum(self):
    if len(self.loss_nodes) != 0:
      raise RuntimeError("There are some uncomputed losses. Call compute() firstly.")
    return sum(loss for loss in self.loss_values.values())

  def __add__(self, other):
    for name, value in self.loss_values.items():
      other.loss_values[name] += value
    return other

  def __len__(self):
    return len(self.loss_values)

  def __repr__(self):
    loss_str = ", ".join(["%s %f" % (loss_name, loss_value) for loss_name, loss_value in self.loss_values.items()])
    return "{Loss Builder: %s}" % (loss_str)
