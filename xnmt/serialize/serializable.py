import yaml

class Serializable(yaml.YAMLObject):
  """
  All model components that appear in a YAML file must inherit from Serializable.
  Implementing classes must specify a unique yaml_tag class attribute, e.g. ``yaml_tag = "!Serializable"``
  """
  def __init__(self):
    # below attributes are automatically set when deserializing (i.e., creating actual objects based on a YAML file)
    # should never be changed manually

    # attributes that are in the YAML file (see Serializable.overwrite_serialize_param() for customizing this)
    self.serialize_params = {}

  def shared_params(self):
    """
    This can be overwritten to specify what parameters of this component and its subcomponents are shared.
    Parameter sharing is performed before any components are initialized, and can therefore only
    include basic data types that are already present in the YAML file (e.g. # dimensions, etc.)
    Sharing is performed if at least one parameter is specified and multiple shared parameters don't conflict.
    In case of conflict a warning is printed, and no sharing is performed.
    The ordering of shared parameters is irrelevant.

    Returns:
      List[Set[Path]]: objects referencing params of this component or a subcompononent
      e.g.::
      
        return [set([Path(".input_dim"),
                     Path(".sub_module.input_dim"),
                     Path(".submodules_list.0.input_dim")])]
    """
    return []

  def overwrite_serialize_param(self, key, val):
    """
    Normally, when serializing an object, the same contents are written as were specified in the config file.
    This method can be called to serialize something else.
    
    Args:
      key (str): name of property
      val (Serializable or basic Python type):
    """
    if not hasattr(self, "serialize_params"):
      self.serialize_params = {}
    self.serialize_params[key] = val

  def __repr__(self):
    if getattr(self, "_is_bare", False):
      return f"bare({self.__class__.__name__}{self._bare_kwargs if self._bare_kwargs else ''})"
    else:
      return f"{self.__class__.__name__}@{id(self)}"


class UninitializedYamlObject(object):
  """
  Wrapper class to indicate an object created by the YAML parser that still needs initialization.
  """
  def __init__(self, data):
    if isinstance(data, UninitializedYamlObject):
      raise AssertionError
    self.data = data
  def get(self, key, default):
    return self.data.get(key, default)

def bare(class_type, **kwargs):
  """
  Returns object of the given class type that looks almost exactly like objects
  created by the YAML parser: object attributes are set, but __init__ has never
  been called.
  
  The main purpose is to enable specifying Serializable subclasses as default
  values. If the subclass were specified as a fully constructed object, this
  will sometimes lead to dependency conflicts, hinder param sharing, etc. when
  deserializing from YAML. Using the ``bare`` mechanism, the YAML deserializer
  can control when exactly the default argument should be constructed.
  
  Args:
    class_type: class type (must be a subclass of :class:`xnmt.serialize.serializable.Serializable`)
    kwargs: will be passed to class's __init__()
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
  A reference to a place in the component hierarchy. Supported a referencing by path or referencing by name.

  Args:
    path (Path): reference-by-path
    name (str): reference-by-name. The name refers to a unique ``_xnmt_id`` property that must be set in exactly one component.
  """
  yaml_tag = "!Ref"

  NO_DEFAULT = 1928437192847

  def __init__(self, path=None, name=None, default=NO_DEFAULT):
    if name is not None and path is not None:
      raise ValueError(f"Ref cannot be initialized with both a name and a path ({name} / {path})")
    self.name = name
    self.path = path
    self.default = default
    self.serialize_params = {'name': name} if name else {'path': str(path)}

  def get_name(self):
    return getattr(self, "name", None)

  def get_path(self):
    return getattr(self, "path", None)

  def is_required(self):
    return getattr(self, "default", Ref.NO_DEFAULT) == Ref.NO_DEFAULT

  def get_default(self):
    return getattr(self, "default", None)

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


class Path(object):
  """
  A relative or absolute path in the component hierarchy.

  Args:
    path_str (str): path string. If prefixed by ".", marks a relative path, otherwise absolute.
  """

  def __init__(self, path_str=""):
    if (len(path_str) > 1 and path_str[-1] == "." and path_str[-2] != ".") \
            or ".." in path_str.strip("."):
      raise ValueError(f"'{path_str}' is not a valid path string")
    self.path_str = path_str

  def append(self, link):
    if not link or "." in link:
      raise ValueError(f"'{link}' is not a valid link")
    if len(self.path_str.strip(".")) == 0:
      return Path(f"{self.path_str}{link}")
    else:
      return Path(f"{self.path_str}.{link}")

  def add_path(self, path_to_add):
    if path_to_add.is_relative_path(): raise NotImplementedError("add_path() is not implemented for relative paths.")
    if len(self.path_str.strip(".")) == 0 or len(path_to_add.path_str) == 0:
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
      if len(s) > 0:
        for link in s.split("."):
          rel_to = rel_to.append(link)
      return rel_to
    else:
      return self

  def descend_one(self):
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
    return self.path_str.split(".")[key]

  def parent(self):
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

  def ancestors(self):
    a = self
    ret = set([a])
    while len(a.path_str.strip(".")) > 0:
      a = a.parent()
      ret.add(a)
    return ret
