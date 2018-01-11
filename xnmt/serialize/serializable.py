import yaml


class Serializable(yaml.YAMLObject):
  """
  All model components that appear in a YAML file must inherit from Serializable.
  Implementing classes must specify a unique yaml_tag class attribute, e.g. yaml_tag = u"!Serializable"
  """
  def __init__(self):
    # below attributes are automatically set when deserializing (i.e., creating actual objects based on a YAML file)
    # should never be changed manually

    # attributes that are in the YAML file (see Serializable.overwrite_serialize_param() for customizing this)
    self.serialize_params = None

    # params passed to __init__, i.e. serialize_params plus shared parameters
    self.init_params = None

  def shared_params(self):
    """
    This can be overwritten to specify what parameters of this component and its subcomponents are shared.
    Parameter sharing is performed before any components are initialized, and can therefore only
    include basic data types that are already present in the YAML file (e.g. # dimensions, etc.)
    Sharing is performed if at least one parameter is specified and multiple shared parameters don't conflict.
    In case of conflict a warning is printed, and no sharing is performed.
    The ordering of shared parameters is irrelevant.

    :returns: list of sets referencing params of this component or a subcompononent
              e.g.:
              return [set(["input_dim", "sub_module.input_dim", submodules_list.0.input_dim"])]
              (the '.0' syntax is available to access elements in a list of subcomponents)
    """
    return []
  def overwrite_serialize_param(self, key, val):
    """
    Overwrites serialize params to something other than specified in the YAML file.
    This is helpful to fix certain model properties (e.g. a vocab) rather than creating it anew
    when serializing and deserializing the component.

    :param key: name of parameter (string)
    :param val: value of parameter (Serializable)
    """
    if not hasattr(self, "serialize_params") or self.serialize_params is None:
      self.serialize_params = {}
    self.serialize_params[key] = val

class UninitializedYamlObject(object):
  """
  Wrapper class to indicate an object created by the YAML parser that still needs initialization.
  """
  def __init__(self, data):
    if isinstance(data, UninitializedYamlObject):
      raise AssertionError
    self.data = data

