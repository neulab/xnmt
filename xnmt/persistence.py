"""
This module takes care of loading and saving YAML files. Both configuration files and saved models are stored in the
same YAML file format.

The main objects to be aware of are:

* :class:`Serializable`: must be subclassed by all components that are specified in a YAML file.
* :class:`Ref`: a reference that points somewhere in the object hierarchy, for both convenience and to realize parameter sharing.
* :class:`Repeat`: a syntax for creating a list components with same configuration but without parameter sharing.
* :class:`YamlPreloader`: pre-loads YAML contents so that some infrastructure can be set up, but does not initialize components.
* :meth:`initialize_if_needed`, :meth:`initialize_object`: initialize a preloaded YAML tree, taking care of resolving references etc.
* :meth:`save_to_file`: saves a YAML file along with registered DyNet parameters
* :class:`LoadSerialized`: can be used to load, modify, and re-assemble pretrained models.
* :meth:`bare`: create uninitialized objects, usually for the purpose of specifying them as default arguments.
* :class:`RandomParam`: a special Serializable subclass that realizes random parameter search.

"""

from functools import singledispatch
from enum import IntEnum, auto
import collections.abc
import numbers
import logging
logger = logging.getLogger('xnmt')
import os
import copy
from functools import lru_cache, wraps
from collections import OrderedDict
import collections.abc
from typing import List, Set, Callable, TypeVar, Type, Union, Optional, Dict, Any
import inspect, random

import yaml

from xnmt import param_collections, tee, utils
import xnmt

def serializable_init(f):
  @wraps(f)
  def wrapper(obj, *args, **kwargs):
    if "xnmt_subcol_name" in kwargs:
      xnmt_subcol_name = kwargs.pop("xnmt_subcol_name")
    elif hasattr(obj, "xnmt_subcol_name"): # happens when calling wrapped super() constructors
      xnmt_subcol_name = obj.xnmt_subcol_name
    else:
      xnmt_subcol_name = _generate_subcol_name(obj)
    obj.xnmt_subcol_name = xnmt_subcol_name
    serialize_params = dict(kwargs)
    params = inspect.signature(f).parameters
    if len(args) > 0:
      param_names = [p.name for p in list(params.values())]
      assert param_names[0] == "self"
      param_names = param_names[1:]
      for i, arg in enumerate(args):
        serialize_params[param_names[i]] = arg
    auto_added_defaults = set()
    for param in params.values():
      if param.name != "self" and param.default != inspect.Parameter.empty and param.name not in serialize_params:
        serialize_params[param.name] = copy.deepcopy(param.default)
        auto_added_defaults.add(param.name)
    for arg in serialize_params.values():
      if type(obj).__name__ != "Experiment":
        assert type(arg).__name__ != "ExpGlobal", \
          "ExpGlobal can no longer be passed directly. Use a reference to its properties instead."
        assert type(arg).__name__ != "ParameterCollection", \
          "cannot pass dy.ParameterCollection to a Serializable class. " \
          "Use ParamManager.my_params() from within the Serializable class's __init__() method instead."
    for key, arg in list(serialize_params.items()):
      if isinstance(arg, Ref):
        if not arg.is_required():
          serialize_params[key] = copy.deepcopy(arg.get_default())
        else:
          if key in auto_added_defaults:
            raise ValueError(
              f"Required argument '{key}' of {type(obj).__name__}.__init__() was not specified, and {arg} could not be resolved")
          else:
            raise ValueError(
              f"Cannot pass a reference as argument; received {serialize_params[key]} in {type(obj).__name__}.__init__()")
    for key, arg in list(serialize_params.items()):
      if getattr(arg, "_is_bare", False):
        initialized = initialize_object(UninitializedYamlObject(arg))
        assert not getattr(initialized, "_is_bare", False)
        serialize_params[key] = initialized
    f(obj, **serialize_params)
    if param_collections.ParamManager.initialized and xnmt_subcol_name in param_collections.ParamManager.param_col.subcols:
      serialize_params["xnmt_subcol_name"] = xnmt_subcol_name
    serialize_params.update(getattr(obj, "serialize_params", {}))
    if "yaml_path" in serialize_params: del serialize_params["yaml_path"]
    obj.serialize_params = serialize_params
    obj.init_completed = True
    # TODO: the below is needed for proper reference creation when saving the model, but should be replaced with
    # something safer
    for key, arg in serialize_params.items():
      if not hasattr(obj, key):
        setattr(obj, key, arg)

  wrapper.uses_serializable_init = True
  return wrapper

class Serializable(yaml.YAMLObject):
  """
  All model components that appear in a YAML file must inherit from Serializable.
  Implementing classes must specify a unique yaml_tag class attribute, e.g. ``yaml_tag = "!Serializable"``
  """
  @serializable_init
  def __init__(self) -> None:
    """
    Initialize class, including allocation of DyNet parameters if needed.

    The __init__() must always be annotated with @serializable_init. It's arguments are exactly those that can be
    specified in a YAML config file. If the argument values are Serializable, they are initialized before being passed
    to this class. The order of the arguments defined here determines in what order children are initialized, which
    may be important when there are dependencies between children.
    """

    # attributes that are in the YAML file (never change this manually, use Serializable.save_processed_arg() instead)
    self.serialize_params = {}

  def shared_params(self) -> List[Set[Union[str,'Path']]]:
    """
    Return the shared parameters of this Serializable class.

    This can be overwritten to specify what parameters of this component and its subcomponents are shared.
    Parameter sharing is performed before any components are initialized, and can therefore only
    include basic data types that are already present in the YAML file (e.g. # dimensions, etc.)
    Sharing is performed if at least one parameter is specified and multiple shared parameters don't conflict.
    In case of conflict a warning is printed, and no sharing is performed.
    The ordering of shared parameters is irrelevant.
    Note also that if a submodule is replaced by a reference, its shared parameters are ignored.

    Returns:
      objects referencing params of this component or a subcompononent
      e.g.::

        return [set([".input_dim",
                     ".sub_module.input_dim",
                     ".submodules_list.0.input_dim"])]
    """
    return []

  def save_processed_arg(self, key: str, val: Any) -> None:
    """
    Save a new value for an init argument (call from within ``__init__()``).

    Normally, the serialization mechanism makes sure that the same arguments are passed when creating the class
    initially based on a config file, and when loading it from a saved model. This method can be called from inside
    ``__init__()`` to save a new value that will be passed when loading the saved model. This can be useful when one
    doesn't want to recompute something every time (like a vocab) or when something has been passed via implicit
    referencing which might yield inconsistent result when loading the model to assemble a new model of different
    structure.

    Args:
      key: name of property, must match an argument of ``__init__()``
      val: new value; a :class:`Serializable` or basic Python type or list or dict of these
    """
    if not hasattr(self, "serialize_params"):
      self.serialize_params = {}
    if key!="xnmt_subcol_name" and key not in _get_init_args_defaults(self):
      raise ValueError(f"{key} is not an init argument of {self}")
    self.serialize_params[key] = val

  def add_serializable_component(self, name: str, passed: Any,
                                 create_fct: Callable[[], Any]) -> Any:
    """
    Create a :class:`Serializable` component, or a container component with several :class:`Serializable`-s.

    :class:`Serializable` sub-components should always be created using this helper to make sure DyNet parameters are
    assigned properly and serialization works properly. The components must also be accepted as init arguments,
    defaulting to ``None``. The helper makes sure that components are only created if ``None`` is passed, otherwise the
    passed component is reused.

    The idiom for using this for an argument named ``my_comp`` would be::

      def __init__(self, my_comp=None, other_args, ...):
        ...
        my_comp = self.add_serializable_component("my_comp", my_comp, lambda: SomeSerializable(other_args))
        # now, do something with my_comp
        ...

    Args:
      name: name of the object
      passed: object as passed in the constructor. If ``None``, will be created using create_fct.
      create_fct: a callable with no arguments that returns a :class:`Serializable` or a collection of
                  :class:`Serializable`-s. When loading a saved model, this same object will be passed via the
                  ``passed`` argument, and ``create_fct`` is not invoked.

    Returns:
      reused or newly created object(s).
    """
    if passed is None:
      initialized = create_fct()
      self.save_processed_arg(name, initialized)
      return initialized
    else:
      return passed

  def __repr__(self):
    if getattr(self, "_is_bare", False):
      return f"bare({self.__class__.__name__}{self._bare_kwargs if self._bare_kwargs else ''})"
    else:
      return f"{self.__class__.__name__}@{id(self)}"


class UninitializedYamlObject(object):
  """
  Wrapper class to indicate an object created by the YAML parser that still needs initialization.

  Args:
    data: uninitialized object
  """

  def __init__(self, data: Any) -> None:
    if isinstance(data, UninitializedYamlObject):
      raise AssertionError
    self.data = data

  def get(self, key: str, default: Any) -> Any:
    return self.data.get(key, default)


T = TypeVar('T')
def bare(class_type: Type[T], **kwargs: Any) -> T:
  """
  Create an uninitialized object of arbitrary type.

  This is useful to specify XNMT components as default arguments. ``__init__()`` commonly requires DyNet parameters,
  component referencing, etc., which are not yet set up at the time the default arguments are loaded.
  In this case, a bare class can be specified with the desired arguments, and will be properly initialized when passed
  as arguments into a component.

  Args:
    class_type: class type (must be a subclass of :class:`Serializable`)
    kwargs: will be passed to class's ``__init__()``
  Returns:
    uninitialized object
  """
  obj = class_type.__new__(class_type)
  assert isinstance(obj, Serializable)
  for key, val in kwargs.items():
    setattr(obj, key, val)
  setattr(obj, "_is_bare", True)
  setattr(obj, "_bare_kwargs", kwargs)
  return obj


class Ref(Serializable):
  """
  A reference to somewhere in the component hierarchy.

  Components can be referenced by path or by name.

  Args:
    path: reference by path
    name: reference by name. The name refers to a unique ``_xnmt_id`` property that must be set in exactly one component.
  """
  yaml_tag = "!Ref"

  NO_DEFAULT = 1928437192847

  @serializable_init
  def __init__(self, path: Union[None, 'Path', str] = None, name: Optional[str] = None,
               default: Any = NO_DEFAULT) -> None:
    if name is not None and path is not None:
      raise ValueError(f"Ref cannot be initialized with both a name and a path ({name} / {path})")
    if isinstance(path, str): path = Path(path)
    self.name = name
    self.path = path
    self.default = default
    self.serialize_params = {'name': name} if name else {'path': str(path)}

  def get_name(self) -> str:
    """Return name, or ``None`` if this is not a named reference"""
    return getattr(self, "name", None)

  def get_path(self) -> Optional['Path']:
    """Return path, or ``None`` if this is a named reference"""
    if getattr(self, "path", None):
      if isinstance(self.path, str): self.path = Path(self.path)
      return self.path
    return None

  def is_required(self) -> bool:
    """Return ``True`` iff there exists no default value and it is mandatory that this reference be resolved."""
    return getattr(self, "default", Ref.NO_DEFAULT) == Ref.NO_DEFAULT

  def get_default(self) -> Any:
    """Return default value, or ``Ref.NO_DEFAULT`` if no default value is set (i.e., this is a required reference)."""
    return getattr(self, "default", None)

  def __str__(self):
    default_str = f", default={self.default}" if getattr(self, "default", Ref.NO_DEFAULT) != Ref.NO_DEFAULT else ""
    if self.get_name():
      return f"Ref(name={self.get_name()}{default_str})"
    else:
      return f"Ref(path={self.get_path()}{default_str})"

  def __repr__(self):
    return str(self)

  def resolve_path(self, named_paths: Dict[str, 'Path']) -> 'Path':
    """Get path, resolving paths properly in case this is a named reference."""
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

class Path(object):
  """
  A relative or absolute path in the component hierarchy.

  Paths are immutable: Operations that change the path always return a new Path object.

  Args:
    path_str: path string, with period ``.`` as separator. If prefixed by ``.``, marks a relative path, otherwise
              absolute.
  """

  def __init__(self, path_str: str = "") -> None:
    if (len(path_str) > 1 and path_str[-1] == "." and path_str[-2] != ".") \
            or ".." in path_str.strip("."):
      raise ValueError(f"'{path_str}' is not a valid path string")
    self.path_str = path_str

  def append(self, link: str) -> 'Path':
    """
    Return a new path by appending a link.

    Args:
      link: link to append

    Returns: new path

    """
    if not link or "." in link:
      raise ValueError(f"'{link}' is not a valid link")
    if len(self.path_str.strip(".")) == 0:
      return Path(f"{self.path_str}{link}")
    else:
      return Path(f"{self.path_str}.{link}")

  def add_path(self, path_to_add: 'Path') -> 'Path':
    """
    Concatenates a path

    Args:
      path_to_add: path to concatenate

    Returns: concatenated path

    """
    if path_to_add.is_relative_path(): raise NotImplementedError("add_path() is not implemented for relative paths.")
    if len(self.path_str.strip(".")) == 0 or len(path_to_add.path_str) == 0:
      return Path(f"{self.path_str}{path_to_add.path_str}")
    else:
      return Path(f"{self.path_str}.{path_to_add.path_str}")

  def __str__(self):
    return self.path_str

  def __repr__(self):
    return self.path_str

  def is_relative_path(self) -> bool:
    return self.path_str.startswith(".")

  def get_absolute(self, rel_to: 'Path') -> 'Path':
    if rel_to.is_relative_path(): raise ValueError("rel_to must be an absolute path!")
    if self.is_relative_path():
      num_up = len(self.path_str) - len(self.path_str.strip(".")) - 1
      for _ in range(num_up):
        rel_to = rel_to.parent()
      s = self.path_str.strip(".")
      if len(s) > 0:
        for link in s.split("."):
          rel_to = rel_to.append(link)
      return rel_to
    else:
      return self

  def descend_one(self) -> 'Path':
    if self.is_relative_path() or len(self) == 0:
      raise ValueError(f"Can't call descend_one() on path {self.path_str}")
    return Path(".".join(self.path_str.split(".")[1:]))

  def __len__(self):
    if self.is_relative_path():
      raise ValueError(f"Can't call __len__() on path {self.path_str}")
    if len(self.path_str) == 0: return 0
    return len(self.path_str.split("."))

  def __getitem__(self, key):
    if self.is_relative_path():
      raise ValueError(f"Can't call __getitem__() on path {self.path_str}")
    if isinstance(key, slice):
      _, _, step = key.indices(len(self))
      if step is not None and step != 1: raise ValueError(f"step must be 1, found {step}")
      return Path(".".join(self.path_str.split(".")[key]))
    else:
      return self.path_str.split(".")[key]

  def parent(self) -> 'Path':
    if len(self.path_str.strip(".")) == 0:
      raise ValueError(f"Path '{self.path_str}' has no parent")
    else:
      spl = self.path_str.split(".")[:-1]
      if '.'.join(spl) == "" and self.path_str.startswith("."):
        return Path(".")
      else:
        return Path(".".join(spl))

  def __hash__(self):
    return hash(self.path_str)

  def __eq__(self, other):
    if isinstance(other, Path):
      return self.path_str == other.path_str
    else:
      return False

  def ancestors(self) -> Set['Path']:
    a = self
    ret = {a}
    while len(a.path_str.strip(".")) > 0:
      a = a.parent()
      ret.add(a)
    return ret

class Repeat(Serializable):
  """
  A special object that is replaced by a list of components with identical configuration but not with shared params.

  This can be specified anywhere in the config hierarchy where normally a list is expected.
  A common use case is a multi-layer neural architecture, where layer configurations are repeated many times.
  It is replaced in the preloader and cannot be instantiated directly.
  """
  yaml_tag = "!Repeat"
  @serializable_init
  def __init__(self, times: numbers.Integral, content: Any) -> None:
    self.times = times
    self.content = content
    raise ValueError("Repeat cannot be instantiated")


_subcol_rand = random.Random()


def _generate_subcol_name(subcol_owner):
  rand_bits = _subcol_rand.getrandbits(32)
  rand_hex = "%008x" % rand_bits
  return f"{type(subcol_owner).__name__}.{rand_hex}"


_reserved_arg_names = ["_xnmt_id", "yaml_path", "serialize_params", "init_params", "kwargs", "self", "xnmt_subcol_name",
                      "init_completed"]

def _get_init_args_defaults(obj):
  return inspect.signature(obj.__init__).parameters


def _check_serializable_args_valid(node):
  base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
  class_param_names = [x[0] for x in inspect.getmembers(node.__class__)]
  init_args = _get_init_args_defaults(node)
  items = {key: val for (key, val) in inspect.getmembers(node)}
  for name in items:
    if name in base_arg_names or name in class_param_names: continue
    if name.startswith("_") or name in _reserved_arg_names: continue
    if name not in init_args:
      raise ValueError(
        f"'{name}' is not a accepted argument of {type(node).__name__}.__init__(). Valid are {list(init_args.keys())}")


@singledispatch
def _name_serializable_children(node):
  return _name_children(node, include_reserved=False)


@_name_serializable_children.register(Serializable)
def _name_serializable_children_serializable(node):
  return getattr(node, "serialize_params", {}).items()


@singledispatch
def _name_children(node, include_reserved):
  return []


@_name_children.register(Serializable)
def _name_children_serializable(node, include_reserved):
  """
  Returns the specified arguments in the order they appear in the corresponding ``__init__()``
  """
  init_args = list(_get_init_args_defaults(node).keys())
  if include_reserved: init_args += [n for n in _reserved_arg_names if not n in init_args]
  items = {key: val for (key, val) in inspect.getmembers(node)}
  ret = []
  for name in init_args:
    if name in items:
      val = items[name]
      ret.append((name, val))
  return ret


@_name_children.register(dict)
def _name_children_dict(node, include_reserved):
  return node.items()


@_name_children.register(list)
def _name_children_list(node, include_reserved):
  return [(str(n), l) for n, l in enumerate(node)]

@_name_children.register(tuple)
def _name_children_tuple(node, include_reserved):
  raise ValueError(f"Tuples are not serializable, use a list instead. Found this tuple: {node}.")

@singledispatch
def _get_child(node, name):
  if not hasattr(node, name): raise PathError(f"{node} has no child named {name}")
  return getattr(node, name)


@_get_child.register(list)
def _get_child_list(node, name):
  try:
    name = int(name)
  except:
    raise PathError(f"{node} has no child named {name} (integer expected)")
  if not 0 <= name < len(node):
    raise PathError(f"{node} has no child named {name} (index error)")
  return node[int(name)]


@_get_child.register(dict)
def _get_child_dict(node, name):
  if not name in node.keys():
    raise PathError(f"{node} has no child named {name} (key error)")
  return node[name]


@_get_child.register(Serializable)
def _get_child_serializable(node, name):
  # if hasattr(node, "serialize_params"):
  #   return _get_child(node.serialize_params, name)
  # else:
    if not hasattr(node, name):
      raise PathError(f"{node} has no child named {name}")
    return getattr(node, name)


@singledispatch
def _set_child(node, name, val):
  pass


@_set_child.register(Serializable)
def _set_child_serializable(node, name, val):
  setattr(node, name, val)


@_set_child.register(list)
def _set_child_list(node, name, val):
  if name == "append": name = len(node)
  try:
    name = int(name)
  except:
    raise PathError(f"{node} has no child named {name} (integer expected)")
  if not 0 <= name < len(node)+1:
    raise PathError(f"{node} has no child named {name} (index error)")
  if name == len(node):
    node.append(val)
  else:
    node[int(name)] = val


@_set_child.register(dict)
def _set_child_dict(node, name, val):
  node[name] = val


def _redirect_path_untested(path, root, cur_node=None):
  # note: this might become useful in the future, but is not carefully tested, use with care
  if cur_node is None: cur_node = root
  if len(path) == 0:
    if isinstance(cur_node, Ref):
      return cur_node.get_path()
    return path
  elif isinstance(cur_node, Ref):
    assert not cur_node.get_path().is_relative_path()
    return _redirect_path_untested(cur_node.get_path(), root, _get_descendant(root, cur_node.get_path()))
  else:
    try:
      return path[:1].add_path(_redirect_path_untested(path.descend_one(), root, _get_child(cur_node, path[0])))
    except PathError:  # child does not exist
      return path


def _get_descendant(node, path, redirect=False):
  if len(path) == 0:
    return node
  elif redirect and isinstance(node, Ref):
    node_path = node.get_path()
    if isinstance(node_path, str): node_path = Path(node_path)
    return Ref(node_path.add_path(path), default=node.get_default())
  else:
    return _get_descendant(_get_child(node, path[0]), path.descend_one(), redirect=redirect)


def _set_descendant(root, path, val):
  if len(path) == 0:
    raise ValueError("path was empty")
  elif len(path) == 1:
    _set_child(root, path[0], val)
  else:
    _set_descendant(_get_child(root, path[0]), path.descend_one(), val)


class _TraversalOrder(IntEnum):
  ROOT_FIRST = auto()
  ROOT_LAST = auto()


def _traverse_tree(node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path(), include_root=True):
  """
  For each node in the tree, yield a (path, node) tuple
  """
  if include_root and traversal_order == _TraversalOrder.ROOT_FIRST:
    yield path_to_node, node
  for child_name, child in _name_children(node, include_reserved=False):
    yield from _traverse_tree(child, traversal_order, path_to_node.append(child_name))
  if include_root and traversal_order == _TraversalOrder.ROOT_LAST:
    yield path_to_node, node


def _traverse_serializable(root, path_to_node=Path()):
  yield path_to_node, root
  for child_name, child in _name_serializable_children(root):
    yield from _traverse_serializable(child, path_to_node.append(child_name))


def _traverse_serializable_breadth_first(root):
  all_nodes = [(path, node) for (path, node) in _traverse_serializable(root)]
  all_nodes = [item[1] for item in sorted(enumerate(all_nodes), key=lambda x: (len(x[1][0]), x[0]))]
  return iter(all_nodes)


def _traverse_tree_deep(root, cur_node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path(), named_paths=None,
                        past_visits=set()):
  """
  Traverse the tree and descend into references. The returned path is that of the resolved reference.

  Args:
    root (Serializable):
    cur_node (Serializable):
    traversal_order (_TraversalOrder):
    path_to_node (Path):
    named_paths (dict):
    past_visits (set):
  """

  # prevent infinite recursion:
  if named_paths is None:
    named_paths = {}
  cur_call_sig = (id(root), id(cur_node), path_to_node)
  if cur_call_sig in past_visits: return
  past_visits = set(past_visits)
  past_visits.add(cur_call_sig)

  if traversal_order == _TraversalOrder.ROOT_FIRST:
    yield path_to_node, cur_node
  if isinstance(cur_node, Ref):
    resolved_path = cur_node.resolve_path(named_paths)
    try:
      yield from _traverse_tree_deep(root, _get_descendant(root, resolved_path), traversal_order, resolved_path,
                                     named_paths, past_visits=past_visits)
    except PathError:
      pass
  else:
    for child_name, child in _name_children(cur_node, include_reserved=False):
      yield from _traverse_tree_deep(root, child, traversal_order, path_to_node.append(child_name), named_paths,
                                     past_visits=past_visits)
  if traversal_order == _TraversalOrder.ROOT_LAST:
    yield path_to_node, cur_node


def _traverse_tree_deep_once(root, cur_node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path(),
                             named_paths=None):
  """
  Calls _traverse_tree_deep, but skips over nodes that have been visited before (can happen because we're descending into
   references).
  """
  if named_paths is None:
    named_paths = {}
  yielded_paths = set()
  for path, node in _traverse_tree_deep(root, cur_node, traversal_order, path_to_node, named_paths):
    if not (path.ancestors() & yielded_paths):
      yielded_paths.add(path)
      yield (path, node)


def _get_named_paths(root):
  d = {}
  for path, node in _traverse_tree(root):
    if "_xnmt_id" in [name for (name, _) in _name_children(node, include_reserved=True)]:
      xnmt_id = _get_child(node, "_xnmt_id")
      if xnmt_id in d:
        raise ValueError(f"_xnmt_id {xnmt_id} was specified multiple times!")
      d[xnmt_id] = path
  return d


class PathError(Exception):
  def __init__(self, message: str) -> None:
    super().__init__(message)


class SavedFormatString(str, Serializable):
  yaml_tag = "!SavedFormatString"
  @serializable_init
  def __init__(self, value: str, unformatted_value: str) -> None:
    self.unformatted_value = unformatted_value
    self.value = value


class FormatString(str, yaml.YAMLObject):
  """
  Used to handle the ``{EXP}`` string formatting syntax.
  When passed around it will appear like the properly resolved string,
  but writing it back to YAML will use original version containing ``{EXP}``
  """

  def __new__(cls, value: str, *args, **kwargs) -> 'FormatString':
    return super().__new__(cls, value)

  def __init__(self, value: str, serialize_as: str) -> None:
    self.value = value
    self.serialize_as = serialize_as

def _init_fs_representer(dumper, obj):
  return dumper.represent_mapping('!SavedFormatString', {"value":obj.value,"unformatted_value":obj.serialize_as})
  # return dumper.represent_data(SavedFormatString(value=obj.value, unformatted_value=obj.serialize_as))
yaml.add_representer(FormatString, _init_fs_representer)

class RandomParam(yaml.YAMLObject):
  yaml_tag = '!RandomParam'

  def __init__(self, values: list) -> None:
    self.values = values

  def __repr__(self):
    return f"{self.__class__.__name__}(values={self.values})"

  def draw_value(self) -> Any:
    if not hasattr(self, 'drawn_value'):
      self.drawn_value = random.choice(self.values)
    return self.drawn_value


class LoadSerialized(Serializable):
  """
  Load content from an external YAML file.

  This object points to an object in an external YAML file and will be replaced by the corresponding content by
  the YAMLPreloader.

  Args:
    filename: YAML file name to load from
    path: path inside the YAML file to load from, with ``.`` separators. Empty string denotes root.
    overwrite: allows overwriting parts of the loaded model with new content. A list of path/val dictionaries, where
               ``path`` is a path string relative to the loaded sub-object following the syntax of :class:`Path`, and
               ``val`` is a Yaml-serializable specifying the new content. E.g.::

                [{"path" : "model.trainer", "val":AdamTrainer()},
                 {"path" : ..., "val":...}]

               It is possible to specify the path to point to a new key to a dictionary.
               If ``path`` points to a list, it's possible append to that list by using ``append_val`` instead of
               ``val``.
  """
  yaml_tag = "!LoadSerialized"

  @serializable_init
  def __init__(self, filename: str, path: str = "", overwrite: Optional[List[Dict[str,Any]]] = None) -> None:
    if overwrite is None: overwrite = []
    self.filename = filename
    self.path = path
    self.overwrite = overwrite

  @staticmethod
  def _check_wellformed(load_serialized):
    _check_serializable_args_valid(load_serialized)
    if hasattr(load_serialized, "overwrite"):
      if not isinstance(load_serialized.overwrite, list):
        raise ValueError(f"LoadSerialized.overwrite must be a list, found: {type(load_serialized.overwrite)}")
      for item in load_serialized.overwrite:
        if not isinstance(item, dict):
          raise ValueError(f"LoadSerialized.overwrite must be a list of dictionaries, found list item: {type(item)}")
        if item.keys() != {"path", "val"}:
          raise ValueError(f"Each overwrite item must have 'path', 'val' (and no other) keys. Found: {item.keys()}")


class YamlPreloader(object):
  """
  Loads experiments from YAML and performs basic preparation, but does not initialize objects.

  Has the following responsibilities:

  * takes care of extracting individual experiments from a YAML file
  * replaces ``!LoadSerialized`` by loading the corresponding content
  * resolves kwargs syntax (items from a kwargs dictionary are moved to the owner where they become object attributes)
  * implements random search (draws proper random values when ``!RandomParam`` is encountered)
  * finds and replaces placeholder strings such as ``{EXP}``, ``{EXP_DIR}``, ``{GIT_REV}``, and ``{PID}``
  * copies bare default arguments into the corresponding objects where appropriate.

  Typically, :meth:`initialize_object` would be invoked by passing the result from the ``YamlPreloader``.
  """

  @staticmethod
  def experiment_names_from_file(filename:str) -> List[str]:
    """Return list of experiment names.

    Args:
      filename: path to YAML file
    Returns:
      experiment names occuring in the given file in lexicographic order.
    """
    try:
      with open(filename) as stream:
        experiments = yaml.load(stream)
    except IOError as e:
      raise RuntimeError(f"Could not read configuration file {filename}: {e}")
    except yaml.constructor.ConstructorError:
      logger.error(
        "for proper deserialization of a class object, make sure the class is a subclass of "
        "xnmt.serialize.serializable.Serializable, specifies a proper yaml_tag with leading '!', and its module is "
        "imported under xnmt/__init__.py")
      raise
    if isinstance(experiments, dict):
      if "defaults" in experiments: del experiments["defaults"]
      return sorted(experiments.keys())
    elif isinstance(experiments, list):
      exp_names = []
      for exp in experiments:
        if not hasattr(exp, "name"): raise ValueError("Encountered unnamed experiment.")
        if exp.name != "default": exp_names.append(exp.name)
      if len(exp_names) != len(set(exp_names)): raise ValueError(f"Found duplicate experiment names: {exp_names}.")
      return exp_names
    else:
      if experiments.__class__.__name__ != "Experiment":
        raise TypeError(f"Top level of config file must be a single Experiment or a list or dict of experiments."
                        f"Found: {experiments} of type {type(experiments)}.")
      if not hasattr(experiments, "name"): raise ValueError("Encountered unnamed experiment.")
      return [experiments.name]

  @staticmethod
  def preload_experiment_from_file(filename: str, exp_name: str, resume: bool = False) -> UninitializedYamlObject:
    """Preload experiment from YAML file.

    Args:
      filename: YAML config file name
      exp_name: experiment name to load
      resume: set to True if we are loading a saved model file directly and want to restore all formatted strings.

    Returns:
      Preloaded but uninitialized object.
    """
    try:
      with open(filename) as stream:
        config = yaml.load(stream)
    except IOError as e:
      raise RuntimeError(f"Could not read configuration file {filename}: {e}")

    if isinstance(config, dict):
      experiment = config[exp_name]
      if getattr(experiment, "name", exp_name) != exp_name:
        raise ValueError(f"Inconsistent experiment name '{exp_name}' / '{experiment.name}'")
      if not isinstance(experiment, LoadSerialized):
        experiment.name = exp_name
    elif isinstance(config, list):
      experiment = None
      for exp in config:
        if not hasattr(exp, "name"): raise ValueError("Encountered unnamed experiment.")
        if exp.name==exp_name: experiment = exp
      if exp is None: raise ValueError(f"No experiment of name '{exp_name}' exists.")
    else:
      experiment = config
      if not hasattr(experiment, "name"): raise ValueError("Encountered unnamed experiment.")
      if not isinstance(experiment, LoadSerialized):
        if experiment.name != exp_name: raise ValueError(f"No experiment of name '{exp_name}' exists.")
    return YamlPreloader.preload_obj(experiment, exp_name=exp_name, exp_dir=os.path.dirname(filename) or ".",
                                     resume=resume)

  @staticmethod
  def preload_obj(root: Any, exp_name: str, exp_dir: str, resume: bool = False) -> UninitializedYamlObject:
    """Preload a given object.

    Preloading a given object, usually an :class:`xnmt.experiment.Experiment` or :class:`LoadSerialized` object as
    parsed by pyyaml, includes replacing ``!LoadSerialized``, resolving ``kwargs`` syntax, and instantiating random
    search.

    Args:
      root: object to preload
      exp_name: experiment name, needed to replace ``{EXP}``
      exp_dir: directory of the corresponding config file, needed to replace ``{EXP_DIR}``
      resume: if True, keep the formatted strings, e.g. set ``{EXP}`` to the value of the previous run if possible

    Returns:
      Preloaded but uninitialized object.
    """
    for _, node in _traverse_tree(root):
      if isinstance(node, Serializable):
        YamlPreloader._resolve_kwargs(node)

    YamlPreloader._copy_duplicate_components(root) # sometimes duplicate objects occur with yaml.load()

    placeholders = {"EXP": exp_name,
                    "PID": os.getpid(),
                    "EXP_DIR": exp_dir,
                    "GIT_REV": tee.get_git_revision()}

    # do this both before and after resolving !LoadSerialized
    root = YamlPreloader._remove_saved_format_strings(root, keep_value=resume)
    YamlPreloader._format_strings(root, placeholders)

    root = YamlPreloader._load_serialized(root)

    random_search_report = YamlPreloader._instantiate_random_search(root)
    if random_search_report:
      setattr(root, 'random_search_report', random_search_report)

    root = YamlPreloader._resolve_repeat(root)

    # if arguments were not given in the YAML file and are set to a bare(Serializable) by default, copy the bare object
    # into the object hierarchy so it can be used w/ param sharing etc.
    YamlPreloader._resolve_bare_default_args(root)

    # do this both before and after resolving !LoadSerialized
    root = YamlPreloader._remove_saved_format_strings(root, keep_value=resume)
    YamlPreloader._format_strings(root, placeholders)

    return UninitializedYamlObject(root)

  @staticmethod
  def _load_serialized(root: Any) -> Any:
    for path, node in _traverse_tree(root, traversal_order=_TraversalOrder.ROOT_LAST):
      if isinstance(node, LoadSerialized):
        LoadSerialized._check_wellformed(node)
        try:
          with open(node.filename) as stream:
            loaded_root = yaml.load(stream)
        except IOError as e:
          raise RuntimeError(f"Could not read configuration file {node.filename}: {e}")
        if os.path.isdir(f"{node.filename}.data"):
          param_collections.ParamManager.add_load_path(f"{node.filename}.data")
        cur_path = Path(getattr(node, "path", ""))
        for _ in range(10):  # follow references
          loaded_trg = _get_descendant(loaded_root, cur_path, redirect=True)
          if isinstance(loaded_trg, Ref):
            cur_path = loaded_trg.get_path()
          else:
            break

        found_outside_ref = True
        self_inserted_ref_ids = set()
        while found_outside_ref:
          found_outside_ref = False
          named_paths = _get_named_paths(loaded_root)
          replaced_paths = {}
          for sub_path, sub_node in _traverse_tree(loaded_trg, path_to_node=cur_path):
            if isinstance(sub_node, Ref) and not id(sub_node) in self_inserted_ref_ids:
              referenced_path = sub_node.resolve_path(named_paths)
              if referenced_path.is_relative_path():
                raise NotImplementedError("Handling of relative paths with LoadSerialized is not yet implemented.")
              if referenced_path in replaced_paths:
                new_ref = Ref(replaced_paths[referenced_path], default=sub_node.get_default())
                _set_descendant(loaded_trg, sub_path[len(cur_path):], new_ref)
                self_inserted_ref_ids.add(id(new_ref))
              # if outside node:
              elif not str(referenced_path).startswith(str(cur_path)):
                found_outside_ref = True
                referenced_obj = _get_descendant(loaded_root, referenced_path)
                _set_descendant(loaded_trg, sub_path[len(cur_path):], referenced_obj)
                # replaced_paths[referenced_path] = sub_path
                replaced_paths[referenced_path] = path.add_path(sub_path[len(cur_path):])
              else:
                new_ref = Ref(path.add_path(referenced_path[len(cur_path):]), default=sub_node.get_default())
                _set_descendant(loaded_trg, sub_path[len(cur_path):], new_ref)
                self_inserted_ref_ids.add(id(new_ref))

        for d in getattr(node, "overwrite", []):
          overwrite_path = Path(d["path"])
          _set_descendant(loaded_trg, overwrite_path, d["val"])
        if len(path) == 0:
          root = loaded_trg
        else:
          _set_descendant(root, path, loaded_trg)
    return root

  @staticmethod
  def _copy_duplicate_components(root):
    obj_ids = set()
    for path, node in _traverse_tree(root, _TraversalOrder.ROOT_LAST):
      if isinstance(node, (list, dict, Serializable)):
        if id(node) in obj_ids:
          _set_descendant(root, path, copy.deepcopy(node))
        obj_ids.add(id(node))

  @staticmethod
  def _resolve_kwargs(obj: Any) -> None:
    """
    If obj has a kwargs attribute (dictionary), set the dictionary items as attributes
    of the object via setattr (asserting that there are no collisions).
    """
    if hasattr(obj, "kwargs"):
      for k, v in obj.kwargs.items():
        if hasattr(obj, k):
          raise ValueError(f"kwargs '{str(k)}' already specified as class member for object '{str(obj)}'")
        setattr(obj, k, v)
      delattr(obj, "kwargs")

  @staticmethod
  def _instantiate_random_search(experiment):
    # TODO: this should probably be refactored: pull out of persistence.py and generalize so other things like
    # grid search and bayesian optimization can be supported
    param_report = {}
    initialized_random_params = {}
    for path, v in _traverse_tree(experiment):
      if isinstance(v, RandomParam):
        if hasattr(v, "_xnmt_id") and v._xnmt_id in initialized_random_params:
          v = initialized_random_params[v._xnmt_id]
        v = v.draw_value()
        if hasattr(v, "_xnmt_id"):
          initialized_random_params[v._xnmt_id] = v
        _set_descendant(experiment, path, v)
        param_report[path] = v
    return param_report

  @staticmethod
  def _resolve_repeat(root):
    for path, node in _traverse_tree(root, traversal_order=_TraversalOrder.ROOT_LAST):
      if isinstance(node, Repeat):
        expanded = []
        for _ in range(node.times):
          expanded.append(copy.deepcopy(node.content))
        if len(path) == 0:
          root = expanded
        else:
          _set_descendant(root, path, expanded)
    return root

  @staticmethod
  def _remove_saved_format_strings(root, keep_value=False):
    for path, node in _traverse_tree(root, traversal_order=_TraversalOrder.ROOT_LAST):
      if isinstance(node, SavedFormatString):
        replace_by = node.value if keep_value else node.unformatted_value
        if len(path) == 0:
          root = replace_by
        else:
          _set_descendant(root, path, replace_by)
    return root

  @staticmethod
  def _resolve_bare_default_args(root: Any) -> None:
    for path, node in _traverse_tree(root):
      if isinstance(node, Serializable):
        init_args_defaults = _get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in _name_children(node, include_reserved=False)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, Serializable) and not isinstance(arg_default, Ref):
              if not getattr(arg_default, "_is_bare", False):
                raise ValueError(
                  f"only Serializables created via bare(SerializableSubtype) are permitted as default arguments; "
                  f"found a fully initialized Serializable: {arg_default} at {path}")
              YamlPreloader._resolve_bare_default_args(arg_default)  # apply recursively
              setattr(node, expected_arg, copy.deepcopy(arg_default))

  @staticmethod
  def _format_strings(root: Any, format_dict: Dict[str, str]) -> None:
    """
    - replaces strings containing ``{EXP}`` and other supported args
    - also checks if there are default arguments for which no arguments are set and instantiates them with replaced
      ``{EXP}`` if applicable
    """
    try:
      format_dict.update(root.exp_global.placeholders)
    except AttributeError:
      pass
    for path, node in _traverse_tree(root):
      if isinstance(node, str):
        try:
          formatted = node.format(**format_dict)
        except (ValueError, KeyError, IndexError):  # will occur e.g. if a vocab entry contains a curly bracket
          formatted = node
        if node != formatted:
          _set_descendant(root,
                          path,
                          FormatString(formatted, node))
      elif isinstance(node, Serializable):
        init_args_defaults = _get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in _name_children(node, include_reserved=False)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, str):
              try:
                formatted = arg_default.format(**format_dict)
              except (ValueError, KeyError):  # will occur e.g. if a vocab entry contains a curly bracket
                formatted = arg_default
              if arg_default != formatted:
                setattr(node, expected_arg, FormatString(formatted, arg_default))


class _YamlDeserializer(object):

  def __init__(self):
    self.has_been_called = False

  def initialize_if_needed(self, obj: Union[Serializable,UninitializedYamlObject]) -> Serializable:
    """
    Initialize if obj has not yet been initialized.

    Note: make sure to always create a new ``_YamlDeserializer`` before calling this, e.g. using
    ``_YamlDeserializer().initialize_object()``

    Args:
      obj: object to be potentially serialized

    Returns:
      initialized object
    """
    if self.is_initialized(obj): return obj
    else: return self.initialize_object(deserialized_yaml_wrapper=obj)

  @staticmethod
  def is_initialized(obj: Union[Serializable,UninitializedYamlObject]) -> bool:
    """
    Returns: ``True`` if a serializable object's ``__init__()`` has been invoked (either programmatically or through
              YAML deserialization).
              ``False`` if ``__init__()`` has not been invoked, i.e. the object has been produced by the YAML parser but
              is not ready to use.
    """
    return type(obj) != UninitializedYamlObject

  def initialize_object(self, deserialized_yaml_wrapper: Any) -> Any:
    """
    Initializes a hierarchy of deserialized YAML objects.

    Note: make sure to always create a new ``_YamlDeserializer`` before calling this, e.g. using
    ``_YamlDeserializer().initialize_object()``

    Args:
      deserialized_yaml_wrapper: deserialized YAML data inside a :class:`UninitializedYamlObject` wrapper (classes are
                                 resolved and class members set, but ``__init__()`` has not been called at this point)
    Returns:
      the appropriate object, with properly shared parameters and ``__init__()`` having been invoked
    """
    assert not self.has_been_called
    self.has_been_called = True
    if self.is_initialized(deserialized_yaml_wrapper):
      raise AssertionError()
    # make a copy to avoid side effects
    self.deserialized_yaml = copy.deepcopy(deserialized_yaml_wrapper.data)
    # make sure only arguments accepted by the Serializable derivatives' __init__() methods were passed
    self.check_args(self.deserialized_yaml)
    # if arguments were not given in the YAML file and are set to a bare(Serializable) by default, copy the bare object into the object hierarchy so it can be used w/ param sharing etc.
    YamlPreloader._resolve_bare_default_args(self.deserialized_yaml)
    self.named_paths = _get_named_paths(self.deserialized_yaml)
    # if arguments were not given in the YAML file and are set to a Ref by default, copy this Ref into the object structure so that it can be properly resolved in a subsequent step
    self.resolve_ref_default_args(self.deserialized_yaml)
    # if references point to places that are not specified explicitly in the YAML file, but have given default arguments, substitute those default arguments
    self.create_referenced_default_args(self.deserialized_yaml)
    # apply sharing as requested by Serializable.shared_params()
    self.share_init_params_top_down(self.deserialized_yaml)
    # finally, initialize each component via __init__(**init_params), while properly resolving references
    initialized = self.init_components_bottom_up(self.deserialized_yaml)
    return initialized

  def check_args(self, root):
    for _, node in _traverse_tree(root):
      if isinstance(node, Serializable):
        _check_serializable_args_valid(node)

  def resolve_ref_default_args(self, root):
    for _, node in _traverse_tree(root):
      if isinstance(node, Serializable):
        init_args_defaults = _get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in _name_children(node, include_reserved=False)]:
            arg_default = copy.deepcopy(init_args_defaults[expected_arg].default)
            if isinstance(arg_default, Ref):
              setattr(node, expected_arg, arg_default)

  def create_referenced_default_args(self, root):
    for path, node in _traverse_tree(root):
      if isinstance(node, Ref):
        referenced_path = node.get_path()
        if not referenced_path:
          continue # skip named paths
        if isinstance(referenced_path, str): referenced_path = Path(referenced_path)
        give_up = False
        for ancestor in sorted(referenced_path.ancestors(), key = lambda x: len(x)):
          try:
            _get_descendant(root, ancestor)
          except PathError:
            try:
              ancestor_parent = _get_descendant(root, ancestor.parent())
              if isinstance(ancestor_parent, Serializable):
                init_args_defaults = _get_init_args_defaults(ancestor_parent)
                if ancestor[-1] in init_args_defaults:
                  referenced_arg_default = init_args_defaults[ancestor[-1]].default
                else:
                  referenced_arg_default = inspect.Parameter.empty
                if referenced_arg_default != inspect.Parameter.empty:
                  _set_descendant(root, ancestor, copy.deepcopy(referenced_arg_default))
              else:
                give_up = True
            except PathError:
              pass
          if give_up: break

  def share_init_params_top_down(self, root):
    abs_shared_param_sets = []
    for path, node in _traverse_tree(root):
      if isinstance(node, Serializable):
        for shared_param_set in node.shared_params():
          shared_param_set = set(Path(p) if isinstance(p, str) else p for p in shared_param_set)
          abs_shared_param_set = set(p.get_absolute(path) for p in shared_param_set)
          added = False
          for prev_set in abs_shared_param_sets:
            if prev_set & abs_shared_param_set:
              prev_set |= abs_shared_param_set
              added = True
              break
          if not added:
            abs_shared_param_sets.append(abs_shared_param_set)
    for shared_param_set in abs_shared_param_sets:
      shared_val_choices = set()
      for shared_param_path in shared_param_set:
        try:
          new_shared_val = _get_descendant(root, shared_param_path)
        except PathError:
          continue
        for _, child_of_shared_param in _traverse_tree(new_shared_val, include_root=False):
          if isinstance(child_of_shared_param, Serializable):
            raise ValueError(f"{path} shared params {shared_param_set} contains Serializable sub-object {child_of_shared_param} which is not permitted")
        if not isinstance(new_shared_val, Ref):
          shared_val_choices.add(new_shared_val)
      if len(shared_val_choices)>1:
        logger.warning(f"inconsistent shared params at {path} for {shared_param_set}: {shared_val_choices}; Ignoring these shared parameters.")
      elif len(shared_val_choices)==1:
        for shared_param_path in shared_param_set:
          try:
            if shared_param_path[-1] in _get_init_args_defaults(_get_descendant(root, shared_param_path.parent())):
              _set_descendant(root, shared_param_path, list(shared_val_choices)[0])
          except PathError:
            pass # can happen when the shared path contained a reference, which we don't follow to avoid unwanted effects

  def init_components_bottom_up(self, root):
    for path, node in _traverse_tree_deep_once(root, root, _TraversalOrder.ROOT_LAST, named_paths=self.named_paths):
      if isinstance(node, Serializable):
        if isinstance(node, Ref):
          hits_before = self.init_component.cache_info().hits
          try:
            resolved_path = node.resolve_path(self.named_paths)
            initialized_component = self.init_component(resolved_path)
          except PathError:
            if getattr(node, "default", Ref.NO_DEFAULT) == Ref.NO_DEFAULT:
              initialized_component = None
            else:
              initialized_component = copy.deepcopy(node.default)
          if self.init_component.cache_info().hits > hits_before:
            logger.debug(f"for {path}: reusing previously initialized {initialized_component}")
        else:
          initialized_component = self.init_component(path)
        if len(path)==0:
          root = initialized_component
        else:
          _set_descendant(root, path, initialized_component)
    return root

  def check_init_param_types(self, obj, init_params):
    for init_param_name in init_params:
      param_sig = _get_init_args_defaults(obj)
      if init_param_name in param_sig:
        annotated_type = param_sig[init_param_name].annotation
        if annotated_type != inspect.Parameter.empty:
          if not check_type(init_params[init_param_name], annotated_type):
            raise ValueError(f"type check failed for '{init_param_name}' argument of {obj}: expected {annotated_type}, received {init_params[init_param_name]} of type {type(init_params[init_param_name])}")

  @lru_cache(maxsize=None)
  def init_component(self, path):
    """
    Args:
      path: path to uninitialized object
    Returns:
      initialized object; this method is cached, so multiple requests for the same path will return the exact same object
    """
    obj = _get_descendant(self.deserialized_yaml, path)
    if not isinstance(obj, Serializable) or isinstance(obj, FormatString):
      return obj
    init_params = OrderedDict(_name_children(obj, include_reserved=False))
    init_args = _get_init_args_defaults(obj)
    if "yaml_path" in init_args: init_params["yaml_path"] = path
    self.check_init_param_types(obj, init_params)
    with utils.ReportOnException({"yaml_path":path}):
      try:
        if hasattr(obj, "xnmt_subcol_name"):
          initialized_obj = obj.__class__(**init_params, xnmt_subcol_name=obj.xnmt_subcol_name)
        else:
          initialized_obj = obj.__class__(**init_params)
        logger.debug(f"initialized {path}: {obj.__class__.__name__}@{id(obj)}({dict(init_params)})"[:1000])
      except TypeError as e:
        raise ComponentInitError(f"An error occurred when calling {type(obj).__name__}.__init__()\n"
                                 f" The following arguments were passed: {init_params}\n"
                                 f" The following arguments were expected: {init_args.keys()}\n"
                                 f" Current path: {path}\n"
                                 f" Error message: {e}")
    return initialized_obj

def _resolve_serialize_refs(root):
  all_serializable = set() # for DyNet param check

  # gather all non-basic types (Serializable, list, dict) in the global dictionary xnmt.resolved_serialize_params
  for _, node in _traverse_serializable(root):
    if isinstance(node, Serializable):
      all_serializable.add(id(node))
      if not hasattr(node, "serialize_params"):
        raise ValueError(f"Cannot serialize node that has no serialize_params attribute: {node}\n"
                         "Did you forget to wrap the __init__() in @serializable_init ?")
      xnmt.resolved_serialize_params[id(node)] = node.serialize_params
    elif isinstance(node, collections.abc.MutableMapping):
      xnmt.resolved_serialize_params[id(node)] = dict(node)
    elif isinstance(node, collections.abc.MutableSequence):
      xnmt.resolved_serialize_params[id(node)] = list(node)

  if not set(id(o) for o in param_collections.ParamManager.param_col.all_subcol_owners) <= all_serializable:
    raise RuntimeError(f"Not all registered DyNet parameter collections written out. "
                       f"Missing: {param_collections.ParamManager.param_col.all_subcol_owners - all_serializable}.\n"
                       f"This indicates that potentially not all components adhere to the protocol of using "
                       f"Serializable.add_serializable_component() for creating serializable sub-components.")

  refs_inserted_at = set()
  refs_inserted_to = set()
  for path_trg, node_trg in _traverse_serializable(root): # loop potential reference targets
    if not refs_inserted_at & path_trg.ancestors(): # skip target if it or its ancestor has already been replaced by a reference
      if isinstance(node_trg, Serializable):
        for path_src, node_src in _traverse_serializable(root): # loop potential nodes that should be replaced by a reference to the current target
          if not path_src in refs_inserted_to: # don't replace by reference if someone is pointing to this node already
            if path_src!=path_trg and node_src is node_trg: # don't reference to self
              # now we're ready to create a reference from node_src to node_trg (node_src will be replaced, node_trg remains unchanged)
              ref = Ref(path=path_trg)
              xnmt.resolved_serialize_params[id(ref)] = ref.serialize_params # make sure the reference itself can be properly serialized

              src_node_parent = _get_descendant(root, path_src.parent())
              src_node_parent_serialize_params = xnmt.resolved_serialize_params[id(src_node_parent)]
              _set_descendant(src_node_parent_serialize_params, Path(path_src[-1]), ref)
              if isinstance(src_node_parent, (collections.abc.MutableMapping, collections.abc.MutableSequence)):
                assert isinstance(_get_descendant(root, path_src.parent().parent()), Serializable), \
                  "resolving references inside nested lists/dicts is not yet implemented"
                src_node_grandparent = _get_descendant(root, path_src.parent().parent())
                src_node_parent_name = path_src[-2]
                xnmt.resolved_serialize_params[id(src_node_grandparent)][src_node_parent_name] = \
                  xnmt.resolved_serialize_params[id(src_node_parent)]
              refs_inserted_at.add(path_src)
              refs_inserted_to.add(path_trg)

def _dump(ser_obj):
  assert len(xnmt.resolved_serialize_params)==0
  _resolve_serialize_refs(ser_obj)
  ret = yaml.dump(ser_obj)
  xnmt.resolved_serialize_params.clear()
  return ret

def save_to_file(fname: str, mod: Any) -> None:
  """
  Save a component hierarchy and corresponding DyNet parameter collection to disk.

  Args:
    fname: Filename to save to.
    mod: Component hierarchy.
  """
  dirname = os.path.dirname(fname)
  if dirname and not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(fname, 'w') as f:
    f.write(_dump(mod))
    param_collections.ParamManager.param_col.save()


def initialize_if_needed(root: Union[Any, UninitializedYamlObject]) -> Any:
  """
  Initialize if obj has not yet been initialized.

  This includes parameter sharing and resolving of references.

  Args:
    root: object to be potentially serialized

  Returns:
    initialized object
  """
  return _YamlDeserializer().initialize_if_needed(root)

def initialize_object(root: UninitializedYamlObject) -> Any:
  """
  Initialize an uninitialized object.

  This includes parameter sharing and resolving of references.

  Args:
    root: object to be serialized

  Returns:
    initialized object
  """
  return _YamlDeserializer().initialize_object(root)


class ComponentInitError(Exception):
  pass


def check_type(obj, desired_type):
  """
  Checks argument types using isinstance, or some custom logic if type hints from the 'typing' module are given.

  Regarding type hints, only a few major ones are supported. This should cover almost everything that would be expected
  in a YAML config file, but might miss a few special cases.
  For unsupported types, this function evaluates to True.
  Most notably, forward references such as 'SomeType' (with apostrophes around the type) are not supported.
  Note  also that typing.Tuple is among the unsupported types because tuples aren't supported by the XNMT serializer.

  Args:
    obj: object whose type to check
    desired_type: desired type of obj

  Returns:
    False if types don't match or desired_type is unsupported, True otherwise.
  """
  try:
    if isinstance(obj, desired_type): return True
    if isinstance(obj, Serializable): # handle some special issues, probably caused by inconsistent imports:
      if obj.__class__.__name__ == desired_type.__name__ or any(
              base.__name__ == desired_type.__name__ for base in obj.__class__.__bases__):
        return True
    return False
  except TypeError:
    if type(desired_type) == str: return True # don't support forward type references
    if desired_type.__class__.__name__ == "_Any":
      return True
    elif desired_type == type(None):
      return obj is None
    elif desired_type.__class__.__name__ == "_Union":
      return any(
        subtype.__class__.__name__ == "_ForwardRef" or check_type(obj, subtype) for subtype in desired_type.__args__)
    elif issubclass(desired_type.__class__, collections.abc.MutableMapping):
      if not isinstance(obj, collections.abc.MutableMapping): return False
      if desired_type.__args__:
        return (desired_type.__args__[0].__class__.__name__ == "_ForwardRef" or all(
          check_type(key, desired_type.__args__[0]) for key in obj.keys())) and (
                         desired_type.__args__[1].__class__.__name__ == "_ForwardRef" or all(
                   check_type(val, desired_type.__args__[1]) for val in obj.values()))
      else: return True
    elif issubclass(desired_type.__class__, collections.abc.Sequence):
      if not isinstance(obj, collections.abc.Sequence): return False
      if desired_type.__args__ and desired_type.__args__[0].__class__.__name__ != "_ForwardRef":
        return all(check_type(item, desired_type.__args__[0]) for item in obj)
      else: return True
    elif desired_type.__class__.__name__ == "TupleMeta":
      if not isinstance(obj, tuple): return False
      if desired_type.__args__:
        if desired_type.__args__[-1] is ...:
          return desired_type.__args__[0].__class__.__name__ == "_ForwardRef" or check_type(obj[0],
                                                                                            desired_type.__args__[0])
        else:
          return len(obj) == len(desired_type.__args__) and all(
            desired_type.__args__[i].__class__.__name__ == "_ForwardRef" or check_type(obj[i], desired_type.__args__[i])
            for i in range(len(obj)))
      else: return True
    return True # case of unsupported types: return True
