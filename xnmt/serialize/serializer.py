import os
import copy
from functools import lru_cache

import yaml

import xnmt.serialize.tree_tools as tree_tools
from xnmt.serialize.serializable import Serializable, UninitializedYamlObject
from xnmt.serialize.tree_tools import get_init_args_defaults, get_descendant

class YamlSerializer(object):
  
  def __init__(self):
    self.representers_added = False

  def initialize_if_needed(self, obj, yaml_context={}):
    if self.is_initialized(obj): return obj
    else: return self.initialize_object(deserialized_yaml_wrapper=obj, yaml_context=yaml_context)

  @staticmethod
  def is_initialized(obj):
    """
    :returns: True if a serializable object's __init__ has been invoked (either programmatically or through YAML deserialization)
              False if __init__ has not been invoked, i.e. the object has been produced by the YAML parser but is not ready to use
    """
    return type(obj) != UninitializedYamlObject

  def initialize_object(self, deserialized_yaml_wrapper, yaml_context={}):
    """
    Initializes a hierarchy of deserialized YAML objects.
    
    :param deserialized_yaml_wrapper: deserialized YAML data inside a UninitializedYamlObject wrapper (classes are resolved and class members set, but __init__() has not been called at this point)
    :param yaml_context: this is passed to __init__ of every created object that expects a argument named yaml_context 
    :returns: the appropriate object, with properly shared parameters and __init__() having been invoked
    """
    self.yaml_context = yaml_context
    if self.is_initialized(deserialized_yaml_wrapper):
      raise AssertionError()
    self.deserialized_yaml = copy.deepcopy(deserialized_yaml_wrapper.data)   # make a copy to avoid side effects
    self.check_args(self.deserialized_yaml)
    self.named_paths = self.get_named_paths(self.deserialized_yaml)
    self.set_serialize_params(self.deserialized_yaml) # sets each component's serialize_params to represent attributes specified in YAML file
    self.resolve_ref_default_args(self.deserialized_yaml)
    self.share_init_params_top_down(self.deserialized_yaml)     # invoke shared_params mechanism, set each component's init_params accordingly
#     # finally, initialize each component via __init__(**init_params)
    return self.init_components_bottom_up(self.deserialized_yaml)
  
  def check_args(self, root):
    for _, node in tree_tools.traverse_tree(root):
      tree_tools.check_serializable_args_valid(node)
  
  def get_named_paths(self, root):
    d = {}
    for path, node in tree_tools.traverse_tree(root):
      if "_xnmt_id" in [name for (name,_) in tree_tools.name_children(node, include_reserved=True)]:
        xnmt_id = tree_tools.get_child(node, "_xnmt_id")
        if xnmt_id in d:
          raise ValueError(f"_xnmt_id {xnmt_id} was specified multiple times!")
        d[xnmt_id] = path
    return d
    
  def set_serialize_params(self, root):
    for _, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
        node.serialize_params = {}
        for name, child in tree_tools.name_children(node):
          node.serialize_params[name] = child
        node.init_params = dict(node.serialize_params)

  def resolve_ref_default_args(self, root):
    for _, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
        init_args_defaults = tree_tools.get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [name for (name,_) in tree_tools.name_children(node)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, tree_tools.Ref):
              setattr(node, expected_arg, arg_default)

  def share_init_params_top_down(self, obj):
    # TODO: this currently fails in some cases where several shared_param_set's specify the same parameters
    # (as was previously the case with the translator's (attender.state_dim, decoder.lstm_dim) and the decoder's (lstm_dim, bridge.dec_dim) )
    for path, node in tree_tools.traverse_tree(obj):
      if isinstance(node, Serializable):
        for shared_param_set in node.shared_params():
          shared_val_choices = set()
          for shared_param_path in shared_param_set:
            try:
              new_shared_val = tree_tools.get_descendant(node, shared_param_path)
            except AttributeError:
              continue
            for _, child_of_shared_param in tree_tools.traverse_tree(new_shared_val, include_root=False):
              if isinstance(child_of_shared_param, Serializable): 
                raise ValueError(f"{path} shared params {shared_param_set} contains Serializable sub-object {child_of_shared_param} which is not permitted")
            shared_val_choices.add(new_shared_val)
          if len(shared_val_choices)>1:
            print(f"WARNING: inconsistent shared params at {path} for {shared_param_set}: {shared_val_choices}; Ignoring these shared parameters.")
          elif len(shared_val_choices)==1:
            for shared_param_path in shared_param_set:
              if shared_param_path[-1] in get_init_args_defaults(get_descendant(node, shared_param_path.parent())):
                tree_tools.set_descendant(node, shared_param_path, list(shared_val_choices)[0])
  
  def init_components_bottom_up(self, root):
    for path, node in tree_tools.traverse_tree_deep_once(root, root, tree_tools.TraversalOrder.ROOT_LAST, named_paths=self.named_paths):
      if isinstance(node, Serializable):
        if isinstance(node, tree_tools.Ref):
          resolved_path = node.resolve_path(self.named_paths)
          hits_before = self.init_component.cache_info().hits
          initialized_component = self.init_component(resolved_path)
          tree_tools.set_descendant(root, path, initialized_component)
          if self.init_component.cache_info().hits > hits_before:
            print(f"reusing previously initialized object at {path}")
          tree_tools.set_descendant(root, path.parent().add("serialize_params").add(path[-1]), initialized_component)
        else:
          tree_tools.set_descendant(root, path, self.init_component(path))
    return root

  @lru_cache(maxsize=None)
  def init_component(self, path):
    """
    :param path: path to uninitialized object
    :returns: initialized object; this method is cached, so multiple requests for the same path will return the exact same object
    """
    obj = tree_tools.get_descendant(self.deserialized_yaml, path)
    init_params = obj.init_params
    init_args = tree_tools.get_init_args_defaults(obj)
    if "yaml_context" in init_args: obj.init_params["yaml_context"] = self.yaml_context
    if "yaml_path" in init_args: obj.init_params["yaml_path"] = path
    serialize_params = obj.serialize_params
    try:
      initialized_obj = obj.__class__(**init_params)
      print(f"initialized {path}: {obj.__class__.__name__}({init_params})")
    except TypeError as e:
      raise ComponentInitError(f"{type(obj)} could not be initialized using params {init_params}, expecting params {init_args.keys()}. "
                               f"Error message: {e}")
    if not hasattr(initialized_obj, "serialize_params"):
      initialized_obj.serialize_params = serialize_params
    return initialized_obj

  @staticmethod
  def init_representer(dumper, obj):
    if not hasattr(obj, "serialize_params"):
      raise RuntimeError(f"Serializing object {obj} that does not possess serialize_params, probably because it was created programmatically, is not possible.")
    if type(obj.serialize_params)==list:
      serialize_params = {param:getattr(obj, param) for param in obj.serialize_params}
    else:
      serialize_params = obj.serialize_params
    return dumper.represent_mapping('!' + obj.__class__.__name__, serialize_params)

  def dump(self, ser_obj):
    if not self.representers_added:
      for SerializableChild in Serializable.__subclasses__():
        yaml.add_representer(SerializableChild, self.init_representer)
      self.representers_added = True
    return yaml.dump(ser_obj)

  def save_to_file(self, fname, mod, persistent_param_collection):
    dirname = os.path.dirname(fname)
    if dirname and not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(fname, 'w') as f:
      f.write(self.dump(mod))
    persistent_param_collection.save(fname + '.data')

  def load_from_file(self, fname, param):
    with open(fname, 'r') as f:
      dict_spec = yaml.load(f)
      corpus_parser = UninitializedYamlObject(dict_spec.corpus_parser)
      model = UninitializedYamlObject(dict_spec.model)
      model_context = UninitializedYamlObject(dict_spec)
    return corpus_parser, model, model_context


class ComponentInitError(Exception):
  pass

