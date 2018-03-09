import inspect
from functools import singledispatch
from enum import IntEnum, auto

import yaml

from xnmt.serialize.serializable import Serializable

class Path(object):
  def __init__(self, path_str=""):
    if (len(path_str)>1 and path_str[-1]=="." and path_str[-2]!=".") \
    or ".." in path_str.strip("."):
      raise ValueError(f"'{path_str}' is not a valid path string")
    self.path_str = path_str
  def append(self, link):
    if not link or "." in link:
      raise ValueError(f"'{link}' is not a valid link")
    if len(self.path_str.strip("."))==0: return Path(f"{self.path_str}{link}")
    else: return Path(f"{self.path_str}.{link}")
  def add_path(self, path_to_add):
    if path_to_add.is_relative_path(): raise NotImplementedError("add_path() is not implemented for relative paths.")
    if len(self.path_str.strip("."))==0 or len(path_to_add.path_str)==0:
      return Path(f"{self.path_str}{path_to_add.path_str}")
    else:
      return Path(f"{self.path_str}.{path_to_add.path_str}")
  def __str__(self):
    return self.path_str
  def __repr__(self):
    return self.path_str
  def is_relative_path(self):
    return self.path_str.startswith(".")
  def get_absolute(self, rel_to):
    if rel_to.is_relative_path(): raise ValueError("rel_to must be an absolute path!")
    if self.is_relative_path():
      num_up = len(self.path_str) - len(self.path_str.strip(".")) - 1
      for _ in range(num_up):
        rel_to = rel_to.parent()
      s = self.path_str.strip(".")
      if len(s)>0:
        for link in s.split("."):
          rel_to = rel_to.append(link)
      return rel_to
    else: return self
  def descend_one(self):
    if self.is_relative_path() or len(self)==0:
      raise ValueError(f"Can't call descend_one() on path {self.path_str}")
    return Path(".".join(self.path_str.split(".")[1:]))
  def __len__(self):
    if self.is_relative_path():
      raise ValueError(f"Can't call __len__() on path {self.path_str}")
    if len(self.path_str)==0: return 0
    return len(self.path_str.split("."))
  def __getitem__(self, key):
    if self.is_relative_path():
      raise ValueError(f"Can't call __getitem__() on path {self.path_str}")
    return self.path_str.split(".")[key]
  def parent(self):
    if len(self.path_str.strip(".")) == 0: raise ValueError(f"Path '{self.path_str}' has no parent")
    else:
      spl = self.path_str.split(".")[:-1]
      if '.'.join(spl)=="" and self.path_str.startswith("."): return Path(".")
      else: return Path(".".join(spl))
  def __hash__(self):
    return hash(self.path_str)
  def __eq__(self, other):
    if isinstance(other,Path):
      return self.path_str == other.path_str
    else:
      return False
  def ancestors(self):
    a = self
    ret = set([a])
    while len(a.path_str.strip("."))>0:
      a = a.parent()
      ret.add(a)
    return ret

class Ref(Serializable):
  yaml_tag = "!Ref"
  def __init__(self, path=None, name=None, required=True):
    if name is not None and path is not None:
      raise ValueError(f"Ref cannot be initialized with both a name and a path ({name} / {path})")
    self.name = name
    self.path = path
    self.required = required
    self.serialize_params = {'name':name} if name else {'path':str(path)}
  def get_name(self):
    return getattr(self, "name", None)
  def get_path(self):
    return getattr(self, "path", None)
  def is_required(self):
    return getattr(self, "required", True)
  def __str__(self):
    if self.get_name():
      return f"Ref(name={self.get_name()})"
    else:
      return f"Ref(path={self.get_path()})"
  def __repr__(self):
    return str(self)
  def resolve_path(self, named_paths):
    if self.get_path():
      if isinstance(self.get_path(), str):
        # need to do this here, because the initializer is never called when
        # Ref objects are specified in the YAML file
        self.path = Path(self.get_path())
      return self.path
    elif self.get_name() in named_paths:
      return named_paths[self.get_name()]
    else:
      raise ValueError(f"Could not resolve path of reference {self}")

reserved_arg_names = ["_xnmt_id", "yaml_path", "serialize_params", "init_params", "kwargs", "self"]

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

def traverse_tree_deep(root, cur_node, traversal_order=TraversalOrder.ROOT_FIRST, path_to_node=Path(), named_paths={}):
  """
  Traverse the tree and descend into references. The returned path is that of the resolved reference.
  """
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
