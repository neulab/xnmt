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
      List[Set[xnmt.serialize.tree_tools.Path]]: objects referencing params of this component or a subcompononent
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
