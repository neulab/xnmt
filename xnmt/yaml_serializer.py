import yaml
import inspect
import datetime
import os

class Serializable(yaml.YAMLObject):
  """
  implementing objects MUST:
  - specify a yaml_tag class attribute
  - call self.save_init_params() at the top of their __init__() 
  """
    
class YamlSerializer(object):
  def __init__(self):
    self.representers_added = False
  
  def init_yaml_objects(self, obj):
    base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
    init_args, _, _, _ = inspect.getargspec(obj.__init__)
    init_args.remove("self")
    kwargs = {}
    for name, val in inspect.getmembers(obj):
      if name in base_arg_names or name.startswith("__"): continue
      if isinstance(val, Serializable):
        kwargs[name] = self.init_yaml_objects(val)
      elif type(val) in [type(None), bool, int, float, str, unicode, datetime.datetime, list, dict, set]:
        kwargs[name] = val
      else:
        continue
      if not name in init_args:
        raise ValueError("unknown init parameter for %s: %s" % (obj.yaml_tag, name))
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
      mod = self.init_yaml_objects(dict_spec)
    param.load_all(fname + '.data')
    return mod
    