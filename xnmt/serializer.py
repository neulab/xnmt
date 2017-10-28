import yaml
import inspect
import datetime
import os
import sys
import six
import copy

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
  def dependent_init_params(self):
    """
    This can be overwritten to share parameters that require dependent components already having been initialized.
    The order of initialization is determined by the order in which components are listed in __init__(),
              and then going bottom-up.
    NOTE: currently only supported for top of component hierarchy
    
    :returns: list of DependentInitParam instances
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

class YamlSerializer(object):
  def __init__(self):
    self.representers_added = False
    self.initialized_shared_components = {}

  def initialize_object(self, deserialized_yaml, yaml_context={}):
    """
    Initializes a hierarchy of deserialized YAML objects.
    
    :param deserialized_yaml: deserialized YAML object (classes are resolved and class members set, but __init__() has not been called at this point)
    :param yaml_context: this is passed to __init__ of every created object that expects a argument named yaml_context 
    :returns: the appropriate object, with properly shared parameters and __init__() having been invoked
    """
    deserialized_yaml = copy.deepcopy(deserialized_yaml)   # make a copy to avoid side effects
    self.set_serialize_params_recursive(deserialized_yaml) # sets each component's serialize_params to represent attributes specified in YAML file
    self.share_init_params_top_down(deserialized_yaml)     # invoke shared_params mechanism, set each component's init_params accordingly
    setattr(deserialized_yaml, "yaml_context", yaml_context)
    # finally, initialize each component via __init__(**init_params)
    return self.init_components_bottom_up(deserialized_yaml, deserialized_yaml.dependent_init_params(), yaml_context=yaml_context)

  def set_serialize_params_recursive(self, obj):
    base_arg_names = map(lambda x: x[0], inspect.getmembers(yaml.YAMLObject))
    if not isinstance(obj, Serializable):
      raise RuntimeError("attempting deserialization of non-Serializable object %s of type %s" % (str(obj), type(obj)))
    init_args = self.get_init_args(obj)
    class_param_names = [x[0] for x in inspect.getmembers(obj.__class__)]
    init_args.remove("self")
    obj.serialize_params = {}
    for name, val in inspect.getmembers(obj):
      if name=="yaml_context":
        raise ValueError("'yaml_context' is a reserved specifier, please rename argument")
      if name in base_arg_names or name.startswith("__") or name in ["serialize_params", "init_params"] or name in class_param_names: continue
      if isinstance(val, Serializable):
        obj.serialize_params[name] = val
        self.set_serialize_params_recursive(val)
      elif type(val) in [type(None), bool, int, float, str, type(six.u("")), datetime.datetime, dict, set]:
        obj.serialize_params[name] = val
      elif type(val)==list:
        obj.serialize_params[name] = val
        for item in val:
          if isinstance(item, Serializable):
            self.set_serialize_params_recursive(item)
      else:
        continue
      if not name in init_args:
        raise ValueError("unknown init parameter for %s: %s" % (obj.yaml_tag, name))
    obj.init_params = dict(obj.serialize_params)

  def share_init_params_top_down(self, obj):
    """
    Sets each component's init_params by extending serialize_params with the shared parameters
    
    :param obj: model hierarchy with prepared serialize_params==init_params
    """
    for shared_params in obj.shared_params():
      val = self.get_val_to_share_or_none(obj, shared_params)
      if val:
        for param_descr in shared_params:
          param_obj, param_name = self.resolve_param_name(obj, param_descr)
          init_args = self.get_init_args(param_obj)
          if param_name in init_args:
            param_obj.init_params[param_name] = val
    for _, val in inspect.getmembers(obj):
      if isinstance(val, Serializable):
        self.share_init_params_top_down(val)

  def get_val_to_share_or_none(self, obj, shared_params):
    val = None
    for param_descr in shared_params:
      param_obj, param_name = self.resolve_param_name(obj, param_descr)
      if not isinstance(param_obj, Serializable):
        raise RuntimeError("Attempting parameter sharing for the non-Serializable "
                            "object %s of type %s, for parent object %s" % (str(param_obj), type(param_obj), str(obj)))
      init_args = self.get_init_args(param_obj)
      if param_name not in init_args: cur_val = None
      else: cur_val = param_obj.init_params.get(param_name, None)
      if cur_val:
        if val is None: val = cur_val
        elif cur_val != val:
          print("WARNING: inconsistent shared params %s" % str(shared_params))
          return None
    return val

  def resolve_param_name(self, obj, param_descr):
    param_obj, param_name = obj, param_descr
    while "." in param_name:
      param_name_spl = param_name.split(".", 1)
      if isinstance(param_obj, list):
        param_obj = param_obj[int(param_name_spl[0])]
      else:
        if not hasattr(param_obj, param_name_spl[0]):
          raise RuntimeError("%s object has no attribute '%s'. Did you forget to specify this in the YAML config?" % (type(param_obj), param_name_spl[0]))
        param_obj = getattr(param_obj, param_name_spl[0])
      param_name = param_name_spl[1]
    return param_obj, param_name

  def init_components_bottom_up(self, obj, dependent_init_params, yaml_context):
    init_params = obj.init_params
    serialize_params = obj.serialize_params
    init_args = self.get_init_args(obj)
    init_args.remove("self")
    for init_arg in init_args:
      if hasattr(obj, init_arg):
        val = getattr(obj, init_arg)
        if isinstance(val, Serializable):
          sub_dependent_init_params = [p.move_down() for p in dependent_init_params if p.matches_component(init_arg)]
          init_params[init_arg] = self.init_components_bottom_up(val, sub_dependent_init_params, yaml_context)
        elif isinstance(val, list):
          sub_dependent_init_params = [p.move_down() for p in dependent_init_params if p.matches_component(init_arg)]
          if len(sub_dependent_init_params) > 0:
            raise Exception("dependent_init_params currently not supported for lists of components")
          new_init_params= []
          for item in val:
            if isinstance(item, Serializable):
              new_init_params.append(self.init_components_bottom_up(item, [], yaml_context))
            else:
              new_init_params.append(item)
          init_params[init_arg] = new_init_params
    for p in dependent_init_params:
      if p.matches_component("") and p.param_name() not in init_params:
        if p.param_name() in init_args:
          init_params[p.param_name()] = p.value_fct()
    if "yaml_context" in init_args: init_params["yaml_context"] = yaml_context # pass yaml_context to constructor if it expects a "yaml_context" argument
    
    initialized_obj = self.reuse_or_init_component(obj, init_params, init_args, serialize_params)

    return initialized_obj
  
  def reuse_or_init_component(self, obj, init_params, init_args, serialize_params):
    """
    :param obj: uninitialized object
    :param init_params: named parameters that should be passed to the object's __init__()
    :param init_args: list of arguments expected by __init__()
    :param serialize_params: serialize_params for the object to be created
    :returns: initialized object (if obj has __xnmt_id and another object with the same
                                  __xnmt_id has been initialized previously, we will
                                  simply return that object, otherwise create it)
    """
    try:
      xnmt_id = getattr(obj, "__xnmt_id", None)
      if xnmt_id and xnmt_id in self.initialized_shared_components:
        initialized_obj = self.initialized_shared_components[xnmt_id]
        print("reusing %s(%s)" % (obj.__class__.__name__, init_params))
      else:
        initialized_obj = obj.__class__(**init_params)
        if xnmt_id:
          self.initialized_shared_components[xnmt_id] = initialized_obj
        print("initialized %s(%s)" % (obj.__class__.__name__, init_params))
    except TypeError as e:
      raise ComponentInitError("%s could not be initialized using params %s, expecting params %s. "
                               "Error message: %s" % (type(obj), init_params, init_args, str(e)))

    if not hasattr(initialized_obj, "serialize_params"):
      initialized_obj.serialize_params = serialize_params
    if xnmt_id:
      initialized_obj.serialize_params["__xnmt_id"] = xnmt_id

    return initialized_obj
  
  def get_init_args(self, obj):
    init_args, _, _, _ = inspect.getargspec(obj.__init__)
    return init_args
  
  @staticmethod
  def init_representer(dumper, obj):
    if type(obj.serialize_params)==list:
      serialize_params = {param:getattr(obj, param) for param in obj.serialize_params}
    else:
      serialize_params = obj.serialize_params
    return dumper.represent_mapping(u'!' + obj.__class__.__name__, serialize_params)
  def dump(self, ser_obj):
    if not self.representers_added:
      for SerializableChild in Serializable.__subclasses__():
        yaml.add_representer(SerializableChild, self.init_representer)
      self.representers_added = True
    return yaml.dump(ser_obj)

  def save_to_file(self, fname, mod, persistent_param_collection):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(fname, 'w') as f:
      f.write(self.dump(mod))
    persistent_param_collection.save(fname + '.data')

  def load_from_file(self, fname, param):
    with open(fname, 'r') as f:
      dict_spec = yaml.load(f)
      corpus_parser = dict_spec.corpus_parser
      model = dict_spec.model
      model_context = dict_spec.model_context
    return corpus_parser, model, model_context

class ComponentInitError(Exception):
  pass

class DependentInitParam(Serializable):
  def __init__(self, param_descr, value_fct):
    self.param_descr = param_descr
    self.value_fct = value_fct
  def move_down(self):
    spl = self.param_descr.split(".")
    assert len(spl)>1
    return DependentInitParam(".".join(spl[1:]), self.value_fct)
  def matches_component(self, candidate_component_name):
    spl = self.param_descr.split(".")
    if candidate_component_name=="": return len(spl)==1
    else: return len(spl)>1 and spl[0] == candidate_component_name
  def param_name(self):
    return self.param_descr.split(".")[-1]
