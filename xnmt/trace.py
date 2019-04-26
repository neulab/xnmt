"""
This modules allows printing a high-level (pseudocode-like) stack trace over neural network functions.

This is achieved by tracing all methods that (a) belong to Serializable classes and (b) return a tt.Tensor, an
ExpressionSequence, or a list with at least one of these objects. The print includes info on tensor dimensions.

The main method to call is trace.print_trace(), a good place to put it would be after the computation of the loss value.
"""


import inspect

import xnmt

class TreeNode(object):
  def __init__(self, name):
    self.name = name
    self.args = []
    self.ret_value = None
    self.subcalls = []
  def __eq__(self, other):
    if other is None: return False
    if not isinstance(other, TreeNode): return False
    if (self.name, self.args, self.ret_value) == (other.name, other.args, other.ret_value):
      if len(self.subcalls) != len(other.subcalls): return False
      else:
        return all(own_subcall==other_subcall for own_subcall,other_subcall in zip(self.subcalls, other.subcalls))

trace_tree = [TreeNode(name="root")]

def reset():
  global trace_tree
  trace_tree = [TreeNode(name="root")]

def str_with_dim(obj):
  if isinstance(obj, list):
    return f"[{', '.join([str_with_dim(item) for item in obj])}]"
  elif isinstance(obj, tuple):
    return f"({', '.join([str_with_dim(item) for item in obj])})"
  elif xnmt.backend_torch and hasattr(obj, "size") and callable(obj.size):
    return f"{obj.__class__.__name__} {tuple(obj.size())}"
  elif hasattr(obj, "dim") and callable(obj.dim):
    return f"{obj.__class__.__name__} {obj.dim()}"
  else:
    return str(obj)

def has_dim(obj):
  if isinstance(obj, list) or isinstance(obj, tuple):
    return any(has_dim(item) for item in obj)
  elif xnmt.backend_torch and hasattr(obj, "size") and callable(obj.size):
    return True
  elif hasattr(obj, "dim") and callable(obj.dim):
    return True
  else:
    return False

def tracing(f):
  sig = inspect.signature(f)
  def do_it(*args, **kwargs):
    new_node = TreeNode(name = f.__name__)
    for ix, param in enumerate(sig.parameters.values()):
      if ix < len(args):
        new_node.args.append((param.name, str_with_dim(args[ix])) if has_dim(args[ix]) else (param.name,))
    trace_tree[-1].subcalls.append(new_node)
    trace_tree.append(new_node)
    result = f(*args, **kwargs)
    new_node.ret_value = str_with_dim(result)
    trace_tree.pop()
    return result
  return do_it

def print_trace(node = None, indent=0):
  if node is None: node = trace_tree[0]
  print(f"{'  ' * indent}{node.name}")
  print(f"{'  ' * indent}    {'; '.join([': '.join(arg) for arg in node.args])}")
  prev = None
  repeats = 0 # TODO: repeat currently limited to single calls, does not capture patterns like (a,b,a,b,a,b,..)
  for subcall in node.subcalls:
    if subcall == prev:
      repeats += 1
    else:
      if repeats > 0:
        print(f"{'  ' * indent}REPEAT {repeats} times")
        repeats = 0
      print_trace(subcall, indent=indent+1)
    prev = subcall
  if repeats > 0:
    print(f"{'  ' * indent}REPEAT {repeats} times")
  print(f"{'  ' * indent}--> {node.name} returns {node.ret_value}")

def make_traceable(obj):
  members = inspect.getmembers(obj)
  for name, member in members:
    if not name.startswith("_") and callable(member):
      try:
        sig = inspect.signature(member)
        # TODO: might also include things like decoder states that encapsulate tensor expressions
        if "xnmt.tensor_tools.Tensor" in str(sig.return_annotation) or "ExpressionSequence" in str(sig.return_annotation):
          setattr(obj, name, tracing(member))
          # from functools.update_wrapper:
          for attr in ('__module__', '__name__', '__qualname__', '__doc__', '__annotations__'):
            try:
              value = getattr(member, attr)
            except AttributeError:
              pass
            else:
              setattr(getattr(obj, name), attr, value)
          for attr in ('__dict__',):
            getattr(getattr(obj, name), attr).update(getattr(member, attr, {}))
          # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
          # from the wrapped function when updating __dict__
          getattr(obj, name).__wrapped__ = member
      except ValueError:
        continue # probably Cython built-in