import inspect
from functools import singledispatch
from enum import IntEnum, auto

import yaml

from xnmt.serialize.serializable import Serializable

class Path(object):
  def __init__(self, *l):
    self.l = [str(i) for i in l]
  @staticmethod
  def from_str(s):
    return Path(*s.split("."))
  def add(self, s):
    return Path(*self.l, *[str(s)])
  def __str__(self):
    return ".".join(self.l)
  def __repr__(self):
    return ".".join(self.l)
  def descend_one(self):
    return Path(*self.l[1:])
  def __len__(self):
    return len(self.l)
  def __getitem__(self, key):
    return self.l[key]
  def parent(self):
    return Path(*self.l[:-1])
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return str(self).__eq__(str(other))


def get_init_args_defaults(obj):
    signature = inspect.signature(obj.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
    }


@singledispatch
def name_children(node, include_reserved=False):
  return []
@name_children.register(Serializable)
def name_children_serializable(node, include_reserved=False):
  """
  Returns the specified arguments in the order they appear in the corresponding __init__() 
  """
  base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
  class_param_names = [x[0] for x in inspect.getmembers(node.__class__)]
  init_args = get_init_args_defaults(node)
  items = {key:val for (key,val) in inspect.getmembers(node)}
  ret = []
  for name in items:
    if name in base_arg_names or name in class_param_names: continue
    if include_reserved:
      if name.startswith("_") and name!="_xnmt_id": continue
    else:
      if name.startswith("_") or name in ["yaml_context", "serialize_params", "init_params", "kwargs"]: continue
    if name in init_args or name=="_xnmt_id":
      val = items[name]
      ret.append((name, val))
    elif name!="_xnmt_id":
      raise ValueError(f"'{name}' is not a valid init parameter of {node}. Valid are {init_args}")
  return ret
@name_children.register(dict)
def name_children_dict(node, include_reserved=False):
  """
  Returns dictionary items in the order specified in the YAML file
  """
  return node.items()
@name_children.register(list)
def name_children_list(node, include_reserved=False):
  return [(str(n),l) for n,l in enumerate(node)]

@singledispatch
def get_child(node, name):
  return getattr(node,name)
@get_child.register(list)
def get_child_list(node, name):
  return node[int(name)]
@get_child.register(dict)
def get_child_dict(node, name):
  return node[name]

@singledispatch
def set_initialized_child(node, name, val):
  pass
@set_initialized_child.register(Serializable)
def set_initialized_child_serializable(node, name, val):
  node.init_params[name] = val
@set_initialized_child.register(list)
def set_initialized_child_list(node, name, val):
  node[int(name)] = val
@set_initialized_child.register(dict)
def set_initialized_child_dict(node, name, val):
  node[name] = val


def get_descendant(node, relative_path):
  if len(relative_path)==0:
    return node
  else:
    return get_descendant(get_child(node, relative_path[0]), relative_path.descend_one())
def set_descendant(node, relative_path, val):
  if len(relative_path)==0:
    raise ValueError("set_descendant() relative_path was empty")
  elif len(relative_path)==1:
    set_initialized_child(node, relative_path[0], val)
  else:
    return set_descendant(get_child(node, relative_path[0]), relative_path.descend_one(), val)

class TraversalOrder(IntEnum):
  ROOT_FIRST = auto()
  ROOT_LAST = auto()
  
def traverse_tree(node, traversal_order=TraversalOrder.ROOT_FIRST, path_to_node=Path(), include_root=True):
  if include_root and traversal_order==TraversalOrder.ROOT_FIRST:
    yield path_to_node, node
  for child_name, child in name_children(node):
    yield from traverse_tree(child, traversal_order, path_to_node.add(child_name))
  if include_root and traversal_order==TraversalOrder.ROOT_LAST:
    yield path_to_node, node
