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
  def shared_params(self):
    """
    :returns: list of tuples referencing a param of this component or a subcompononent;
              param values to be shared are determined if at least one parameter is specified and multiple parameters don't conflict.
              in this case, the determined value is copied over to the unspecified parameters
    """
    return []
  def dependent_init_params(self):
    """
    :returns: list of DependentInitParam; these are resolved right before the corresponding model is initialized,
              thereby assuming that the components it depends on have already been initialized.
              The order of initialization is determined by the order in which components are listed in __init__(),
              and then going bottom-up
    """
    return []    
class YamlSerializer(object):
  def __init__(self):
    self.representers_added = False
  
  def initialize_object(self, deserialized_yaml, context={}):
    """
    :param obj: deserialized YAML object (classes are resolved and class members set, but __init__() has not been called at this point)
    :returns: the appropriate object, with properly shared parameters and __init__() having been invoked 
    """
    self.set_serialize_params_recursive(deserialized_yaml)
    self.share_init_params_top_down(deserialized_yaml)
    setattr(deserialized_yaml, "context", context)
    return self.init_components_bottom_up(deserialized_yaml, deserialized_yaml.dependent_init_params())
    
  def set_serialize_params_recursive(self, obj):
    base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
    init_args, _, _, _ = inspect.getargspec(obj.__init__)
    class_param_names = [x[0] for x in inspect.getmembers(obj.__class__)]
    init_args.remove("self")
    obj.serialize_params = {}
    for name, val in inspect.getmembers(obj):
      if name in base_arg_names or name.startswith("__") or name in ["serialize_params", "init_params"] or name in class_param_names: continue
      if isinstance(val, Serializable):
        obj.serialize_params[name] = val
        self.set_serialize_params_recursive(val)
      elif type(val) in [type(None), bool, int, float, str, unicode, datetime.datetime, list, dict, set]:
        obj.serialize_params[name] = val
      else:
        continue
      if not name in init_args:
        raise ValueError("unknown init parameter for %s: %s" % (obj.yaml_tag, name))
    obj.init_params = dict(obj.serialize_params)
    
  def share_init_params_top_down(self, obj):
    """
    sets each component's init_params by extending serialize_params with the shared parameters
    :param obj: model hierarchy with prepared serialize_params=init_params
    """
    for shared_params in obj.shared_params():
      val = self.get_val_to_share_or_none(obj, shared_params)
      if val:
        for param_descr in shared_params:
          param_obj, param_name = self.resolve_param_name(obj, param_descr)
          param_obj.init_params[param_name] = val
    for _, val in inspect.getmembers(obj):
      if isinstance(val, Serializable):
        self.share_init_params_top_down(val)
  def get_val_to_share_or_none(self, obj, shared_params):
    val = None
    for param_descr in shared_params:
      param_obj, param_name = self.resolve_param_name(obj, param_descr)
      init_args, _, _, _ = inspect.getargspec(param_obj.__init__)
      if param_name not in init_args: raise ValueError("unknown init parameter for %s: %s" % (param_obj.yaml_tag, param_name))
      cur_val = param_obj.init_params.get(param_name, None)
      if cur_val:
        if val is None: val = cur_val
        elif cur_val != val:
          print "WARNING: inconsistent shared params %s" % str(shared_params)
          return None
    return val
  def resolve_param_name(self, obj, param_descr):
    param_obj, param_name = obj, param_descr
    while "." in param_name:
      param_name_spl = param_name.split(".", 1)
      param_obj = getattr(param_obj, param_name_spl[0])
      param_name = param_name_spl[1]
    return param_obj, param_name

  def init_components_bottom_up(self, obj, post_init_shared_params):
    init_params = obj.init_params
    serialize_params = obj.serialize_params
    init_args, _, _, _ = inspect.getargspec(obj.__init__)
    for init_arg in init_args:
      if hasattr(obj, init_arg):
        val = getattr(obj, init_arg)
        if isinstance(val, Serializable):
          sub_post_init_shared_params = [p.move_down() for p in post_init_shared_params if p.matches_component(init_arg)]
          init_params[init_arg] = self.init_components_bottom_up(val, sub_post_init_shared_params)
    for p in post_init_shared_params:
      if p.component_name == "." and p.param_name not in init_params:
        init_params[p.param_name] = p.value_fct()
    init_params = {key:val for key,val in init_params.items() if key in init_args}
    print(type(obj))
    initialized_obj = obj.__class__(**init_params)
    if not hasattr(initialized_obj, "serialize_params"):
      initialized_obj.serialize_params = serialize_params
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
    params.save(fname + '.data')
    
  def load_from_file(self, fname, param, context={}):
    with open(fname, 'r') as f:
      dict_spec = yaml.load(f)
      corpus_parser = self.initialize_object(dict_spec.corpus_parser)
      model = self.initialize_object(dict_spec.model, context=context)
    try: # dynet v2
      param.populate(fname + '.data')
    except AttributeError: # dynet v1
      param.load_all(fname + '.data')
    return corpus_parser, model
    
class DependentInitParam(object):
  def __init__(self, component_name, param_name, value_fct):
    self.component_name = component_name + ("" if component_name.endswith(".") else ".")
    self.param_name = param_name
    self.value_fct = value_fct
  def move_down(self):
    return DependentInitParam(self.component_name.split(".", 1)[1], self.param_name, self.value_fct)
  def matches_component(self, candidate_componenet_name):
    return self.component_name.split(".")[0] == candidate_componenet_name
