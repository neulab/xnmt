import inspect
from functools import singledispatch
from enum import IntEnum, auto

import yaml

from xnmt.serialize.serializable import Serializable, Path, Ref


reserved_arg_names = ["_xnmt_id", "yaml_path", "serialize_params", "init_params", "kwargs", "self", "xnmt_subcol_name"]

def get_init_args_defaults(obj):
    return inspect.signature(obj.__init__).parameters

def check_serializable_args_valid(node):
  base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
  class_param_names = [x[0] for x in inspect.getmembers(node.__class__)]
  init_args = get_init_args_defaults(node)
  items = {key:val for (key,val) in inspect.getmembers(node)}
  for name in items:
    if name in base_arg_names or name in class_param_names: continue
    if name.startswith("_") or name in reserved_arg_names: continue
    if name not in init_args:
      raise ValueError(f"'{name}' is not a valid init parameter of {node}. Valid are {list(init_args.keys())}")


@singledispatch
def name_serializable_children(node):
  return name_children(node, include_reserved=False)
@name_serializable_children.register(Serializable)
def name_serializable_children_serializable(node):
  return getattr(node, "serialize_params", {}).items()

@singledispatch
def name_children(node, include_reserved):
  return []
@name_children.register(Serializable)
def name_children_serializable(node, include_reserved):
  """
  Returns the specified arguments in the order they appear in the corresponding __init__()
  """
  init_args = list(get_init_args_defaults(node).keys())
  if include_reserved: init_args += [n for n in reserved_arg_names if not n in init_args]
  items = {key:val for (key,val) in inspect.getmembers(node)}
  ret = []
  for name in init_args:
    if name in items:
      val = items[name]
      ret.append((name, val))
  return ret
@name_children.register(dict)
def name_children_dict(node, include_reserved):
  return node.items()
@name_children.register(list)
def name_children_list(node, include_reserved):
  return [(str(n),l) for n,l in enumerate(node)]

@singledispatch
def get_child(node, name):
  if not hasattr(node, name): raise PathError(f"{node} has not child named {name}")
  return getattr(node,name)
@get_child.register(list)
def get_child_list(node, name):
  try:
    name = int(name)
  except:
    raise PathError(f"{node} has not child named {name} (integer expected)")
  if not 0 <= name < len(node):
    raise PathError(f"{node} has not child named {name} (index error)")
  return node[int(name)]
@get_child.register(dict)
def get_child_dict(node, name):
  if not name in node.keys():
    raise PathError(f"{node} has not child named {name} (key error)")
  return node[name]
@get_child.register(Serializable)
def get_child_serializable(node, name):
  if not hasattr(node, name):
    raise PathError(f"{node} has not child named {name}")
  return getattr(node,name)

@singledispatch
def set_child(node, name, val):
  pass
@set_child.register(Serializable)
def set_child_serializable(node, name, val):
  setattr(node,name,val)
@set_child.register(list)
def set_child_list(node, name, val):
  try:
    name = int(name)
  except:
    raise PathError(f"{node} has not child named {name} (integer expected)")
  if not 0 <= name < len(node):
    raise PathError(f"{node} has not child named {name} (index error)")
  node[int(name)] = val
@set_child.register(dict)
def set_child_dict(node, name, val):
  node[name] = val

def get_descendant(node, path):
  if len(path)==0:
    return node
  else:
    return get_descendant(get_child(node, path[0]), path.descend_one())
def set_descendant(root, path, val):
  if len(path)==0:
    raise ValueError("path was empty")
  elif len(path)==1:
    set_child(root, path[0], val)
  else:
    set_descendant(get_child(root, path[0]), path.descend_one(), val)

class TraversalOrder(IntEnum):
  ROOT_FIRST = auto()
  ROOT_LAST = auto()

def traverse_tree(node, traversal_order=TraversalOrder.ROOT_FIRST, path_to_node=Path(), include_root=True):
  """
  For each node in the tree, yield a (path, node) tuple
  """
  if include_root and traversal_order==TraversalOrder.ROOT_FIRST:
    yield path_to_node, node
  for child_name, child in name_children(node, include_reserved=False):
    yield from traverse_tree(child, traversal_order, path_to_node.append(child_name))
  if include_root and traversal_order==TraversalOrder.ROOT_LAST:
    yield path_to_node, node

def traverse_serializable(root, path_to_node=Path()):
  yield path_to_node, root
  for child_name, child in name_serializable_children(root):
    yield from traverse_serializable(child, path_to_node.append(child_name))
 
def traverse_serializable_breadth_first(root):
  all_nodes = [(path,node) for (path,node) in traverse_serializable(root)]
  all_nodes = [item[1] for item in sorted(enumerate(all_nodes), key=lambda x: (len(x[1][0]),x[0]))]
  return iter(all_nodes)

def traverse_tree_deep(root, cur_node, traversal_order=TraversalOrder.ROOT_FIRST, path_to_node=Path(), named_paths={}, past_visits=set()):
  """
  Traverse the tree and descend into references. The returned path is that of the resolved reference.

  args:
    root (Serializable):
    cur_node (Serializable):
    traversal_order (TraversalOrder):
    path_to_node (Path):
    name_paths (dict):
    past_visits (set):
  """

  # prevent infinite recursion:
  cur_call_sig = (id(root), id(cur_node), path_to_node)
  if cur_call_sig in past_visits: return
  past_visits.add(cur_call_sig)

  if traversal_order==TraversalOrder.ROOT_FIRST:
    yield path_to_node, cur_node
  if isinstance(cur_node, Ref):
    resolved_path = cur_node.resolve_path(named_paths)
    try:
      yield from traverse_tree_deep(root, get_descendant(root, resolved_path), traversal_order, resolved_path, named_paths)
    except PathError:
      if cur_node.is_required():
        raise ValueError(f"Was not able to find required reference '{resolved_path}' at '{path_to_node}'")
  else:
    for child_name, child in name_children(cur_node, include_reserved=False):
      yield from traverse_tree_deep(root, child, traversal_order, path_to_node.append(child_name), named_paths)
  if traversal_order==TraversalOrder.ROOT_LAST:
    yield path_to_node, cur_node

def traverse_tree_deep_once(root, cur_node, traversal_order=TraversalOrder.ROOT_FIRST, path_to_node=Path(), named_paths={}):
  """
  Calls traverse_tree_deep, but skips over nodes that have been visited before (can happen because we're descending into references).
  """
  yielded_paths = set()
  for path, node in traverse_tree_deep(root, cur_node, traversal_order, path_to_node, named_paths):
    if not (path.ancestors() & yielded_paths):
      yielded_paths.add(path)
      yield (path, node)

class PathError(Exception):
  def __init__(self, message):
    super().__init__(message)
