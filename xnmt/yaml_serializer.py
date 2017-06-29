import yaml
import inspect
import datetime
import os

class Serializable(yaml.YAMLObject):
  """
  implementing classes MUST specify a unique yaml_tag class attribute
  """
  def __init__(self):
    self.serialize_params = None # parameters that are in the YAML file
    self.init_params = None # params passed to __init__, i.e. serialize_params plus shared parameters
    
class YamlSerializer(object):
  def __init__(self):
    self.representers_added = False
  
  def create_model(self, deserialized_yaml_model):
    """
    :param obj: deserialized YAML object (classes are resolved and class members set, but __init__() has not been called at this point)
    :returns: models, with properly shared parameters and __init__() having been invoked 
    """
    self.set_serialize_params_recursive(deserialized_yaml_model)
    self.share_init_params(deserialized_yaml_model)
    return self.init_components_bottom_up(deserialized_yaml_model)
    
  def set_serialize_params_recursive(self, obj):
    base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
    init_args, _, _, _ = inspect.getargspec(obj.__init__)
    init_args.remove("self")
    obj.serialize_params = {}
    for name, val in inspect.getmembers(obj):
      if name in base_arg_names or name.startswith("__") or name in ["serialize_params", "init_params"]: continue
      if isinstance(val, Serializable):
        obj.serialize_params[name] = val
        self.set_serialize_params_rec(val)
      elif type(val) in [type(None), bool, int, float, str, unicode, datetime.datetime, list, dict, set]:
        obj.serialize_params[name] = val
      else:
        continue
      if not name in init_args:
        raise ValueError("unknown init parameter for %s: %s" % (obj.yaml_tag, name))
  def share_init_params(self, obj):
    """
    sets each parameters init_params by extending serialize_params with the shared parameters
    :param obj: model hierarchy with serialize_params set
    """
    for _, val in inspect.getmembers(obj):
      if isinstance(val, Serializable):
        self.share_init_params(val)
    setattr(obj, "init_params", obj.serialize_params)
    # TODO: implement actual sharing
  def init_components_bottom_up(self, obj):
    kwargs = obj.init_params
    for name, val in inspect.getmembers(obj):
      if isinstance(val, Serializable):
        kwargs[name] = self.init_components_bottom_up(val)
    initialized_obj = obj.__class__(**kwargs)
    initialized_obj.serialize_params = kwargs
    return initialized_obj
  
  @staticmethod
  def init_representer(dumper, obj):
    return dumper.represent_mapping(u'!' + obj.__class__.__name__, obj.serialize_params)
  def dump(self, ser_obj):
    if not self.representers_added:
      for SerializableChild in Serializable.__subclasses__():
        yaml.add_representer(SerializableChild, self.init_representer)
      self.representers_added = True
    return yaml.dump(ser_obj)

  def save_to_file(self, fname, mod, params):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(fname, 'w') as f:
      f.write(self.dump(mod))
    params.save_all(fname + '.data')
    
  def load_from_file(self, fname, param):
    with open(fname, 'r') as f:
      dict_spec = yaml.load(f)
      mod = self.create_model(dict_spec)
    param.load_all(fname + '.data')
    return mod
    