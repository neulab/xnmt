
import dynet
import collections

class LossBuilder(object):
  def __init__(self):
    self.computed = True
    self.loss_nodes  = []
    self.loss_values = collections.defaultdict(float)

  def add_node(self, loss_func, loss_args=[], loss_kwargs={}):
    self.computed = False
    loss_name = format_loss_name(repr(loss_func))
    loss_graph = loss_func(*loss_args, **loss_kwargs)
    self.loss_nodes.append((loss_name, loss_graph))

  def compute(self):
    ''' Compute all the losses and delete the computational graph reference.
    '''
    self.computed = True
    total_loss  = None
    for loss_name, loss_graph in self.loss_nodes:
      if loss_graph is None:
        continue

      # TODO(philip30): Handle unbatched version here?
      value = loss_graph.npvalue()
      loss_graph = dynet.sum_batches(loss_graph)
      value = value.sum()

      if total_loss is not None:
        total_loss += loss_graph
      else:
        total_loss = loss_graph
      self.loss_values[loss_name] += value

    self.loss_nodes.clear()
    return total_loss

  def __getitem__(self, index):
    return self.loss_nodes[index][1]

  def __iter__(self):
    return iter(self.loss_values.items())

  def sum(self):
    if not self.computed:
      raise RuntimeError("There are some uncomputed losses. Call compute() firstly.")
    return sum(loss for loss in self.loss_values.values())

  def __add__(self, other):
    for name, value in self.loss_values.items():
      other.loss_values[name] += value
    return other

  def __repr__(self):
    loss_str = ", ".join(["%s %f" % (loss_name, loss_value) for loss_name, loss_value in self.loss_values.items()])
    return "{Loss Builder: %s}" % (loss_str)

def format_loss_name(loss_name):
  cols = loss_name.split()
  return cols[2]
