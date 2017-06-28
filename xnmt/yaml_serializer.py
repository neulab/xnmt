import yaml
import inspect
import datetime

class Serializable(yaml.YAMLObject):
  """
  implementing objects must specify a yaml_tag class attribute and a properly set __repr__ method
  TODO: maybe both can be implemented in a general way in this parent class?
  """
  pass

def init_yaml_objects(obj):
  base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
  init_args, _, _, _ = inspect.getargspec(obj.__init__)
  init_args.remove("self")
  kwargs = {}
  for name, val in inspect.getmembers(obj):
    if name in base_arg_names or name.startswith("__"): continue
    if isinstance(val, Serializable):
      kwargs[name] = init_yaml_objects(val)
    elif type(val) in [type(None), bool, int, float, str, unicode, datetime.datetime, list, dict, set]:
      kwargs[name] = val
    else:
      continue
    if not name in init_args:
      raise ValueError("unknown init parameter for %s: %s" % (obj.yaml_tag, name))
  return obj.__class__(**kwargs)