from functools import wraps
import logging
logger = logging.getLogger('xnmt')
import os
import copy
from functools import lru_cache
from collections import OrderedDict
import inspect

import yaml

import xnmt.serialize.tree_tools as tree_tools
from xnmt.serialize.serializable import Serializable, UninitializedYamlObject, Ref

class YamlSerializer(object):

  def initialize_if_needed(self, obj):
    if self.is_initialized(obj): return obj
    else: return self.initialize_object(deserialized_yaml_wrapper=obj)

  @staticmethod
  def is_initialized(obj):
    """
    Returns: True if a serializable object's __init__ has been invoked (either programmatically or through YAML deserialization)
              False if __init__ has not been invoked, i.e. the object has been produced by the YAML parser but is not ready to use
    """
    return type(obj) != UninitializedYamlObject

  def initialize_object(self, deserialized_yaml_wrapper):
    """
    Initializes a hierarchy of deserialized YAML objects.

    Args:
      deserialized_yaml_wrapper: deserialized YAML data inside a UninitializedYamlObject wrapper (classes are resolved and class members set, but __init__() has not been called at this point)
    Returns:
      the appropriate object, with properly shared parameters and __init__() having been invoked
    """
    if self.is_initialized(deserialized_yaml_wrapper):
      raise AssertionError()
    # make a copy to avoid side effects
    self.deserialized_yaml = copy.deepcopy(deserialized_yaml_wrapper.data)
    # make sure only arguments accepted by the Serializable derivatives' __init__() methods were passed
    self.check_args(self.deserialized_yaml)
    self.named_paths = self.get_named_paths(self.deserialized_yaml)
    # if arguments were not given in the YAML file and are set to a Ref by default, copy this Ref into the object structure so that it can be properly resolved in a subsequent step
    self.resolve_ref_default_args(self.deserialized_yaml)
    # if references point to places that are not specified explicitly in the YAML file, but have given default arguments, substitute those default arguments
    self.create_referenced_default_args(self.deserialized_yaml)
    # apply sharing as requested by Serializable.shared_params()
    self.share_init_params_top_down(self.deserialized_yaml)
    # finally, initialize each component via __init__(**init_params), while properly resolving references
    return self.init_components_bottom_up(self.deserialized_yaml)

  def check_args(self, root):
    for _, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
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

  def resolve_ref_default_args(self, root):
    for _, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
        init_args_defaults = tree_tools.get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in tree_tools.name_children(node, include_reserved=False)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, tree_tools.Ref):
              setattr(node, expected_arg, arg_default)

  def create_referenced_default_args(self, root):
    for _, node in tree_tools.traverse_tree(root):
      if isinstance(node, tree_tools.Ref):
        referenced_path = node.get_path()
        if not referenced_path:
          continue # skip named paths
        if isinstance(referenced_path, str): referenced_path = tree_tools.Path(referenced_path)
        give_up = False
        for ancestor in sorted(referenced_path.ancestors(), key = lambda x: len(x)):
          try:
            tree_tools.get_descendant(root, ancestor)
          except tree_tools.PathError:
            try:
              ancestor_parent = tree_tools.get_descendant(root, ancestor.parent())
              if isinstance(ancestor_parent, Serializable):
                init_args_defaults = tree_tools.get_init_args_defaults(ancestor_parent)
                if ancestor[-1] in init_args_defaults:
                  referenced_arg_default = init_args_defaults[ancestor[-1]].default
                else:
                  referenced_arg_default = inspect.Parameter.empty
                if referenced_arg_default == inspect.Parameter.empty:
                  if node.is_required():
                    raise ValueError(f"Reference '{node}' is required but does not exist and has no default arguments")
                else:
                  tree_tools.set_descendant(root, ancestor, referenced_arg_default)
              else:
                if node.is_required():
                  raise ValueError(f"Reference '{node}' is required but does not exist")
                give_up = True
            except tree_tools.PathError:
              if node.is_required():
                raise ValueError(f"Reference '{node}' is required but does not exist")
          if give_up: break

  def share_init_params_top_down(self, root):
    abs_shared_param_sets = []
    for path, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
        for shared_param_set in node.shared_params():
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
          new_shared_val = tree_tools.get_descendant(root, shared_param_path)
        except tree_tools.PathError:
          continue
        for _, child_of_shared_param in tree_tools.traverse_tree(new_shared_val, include_root=False):
          if isinstance(child_of_shared_param, Serializable):
            raise ValueError(f"{path} shared params {shared_param_set} contains Serializable sub-object {child_of_shared_param} which is not permitted")
        shared_val_choices.add(new_shared_val)
      if len(shared_val_choices)>1:
        logger.warning(f"inconsistent shared params at {path} for {shared_param_set}: {shared_val_choices}; Ignoring these shared parameters.")
      elif len(shared_val_choices)==1:
        for shared_param_path in shared_param_set:
          if shared_param_path[-1] in tree_tools.get_init_args_defaults(tree_tools.get_descendant(root, shared_param_path.parent())):
            tree_tools.set_descendant(root, shared_param_path, list(shared_val_choices)[0])

  def init_components_bottom_up(self, root):
    for path, node in tree_tools.traverse_tree_deep_once(root, root, tree_tools.TraversalOrder.ROOT_LAST, named_paths=self.named_paths):
      if isinstance(node, Serializable):
        if isinstance(node, tree_tools.Ref):
          try:
            resolved_path = node.resolve_path(self.named_paths)
            hits_before = self.init_component.cache_info().hits
            initialized_component = self.init_component(resolved_path)
          except tree_tools.PathError:
            initialized_component = None
          if self.init_component.cache_info().hits > hits_before:
            logger.debug(f"for {path}: reusing previously initialized {initialized_component}")
        else:
          initialized_component = self.init_component(path)
        if len(path)==0:
          root = initialized_component
        else:
          tree_tools.set_descendant(root, path, initialized_component)
    return root

  @lru_cache(maxsize=None)
  def init_component(self, path):
    """
    Args:
      path: path to uninitialized object
    Returns:
      initialized object; this method is cached, so multiple requests for the same path will return the exact same object
    """
    obj = tree_tools.get_descendant(self.deserialized_yaml, path)
    if not isinstance(obj, Serializable):
      return obj
    init_params = OrderedDict(tree_tools.name_children(obj, include_reserved=False))
    init_args = tree_tools.get_init_args_defaults(obj)
    if "yaml_path" in init_args: init_params["yaml_path"] = path
    try:
      initialized_obj = obj.__class__(**init_params)
      logger.debug(f"initialized {path}: {obj.__class__.__name__}@{id(obj)}({dict(init_params)})"[:1000])
    except TypeError as e:
      raise ComponentInitError(f"{type(obj)} could not be initialized using params {init_params}, expecting params {init_args.keys()}. "
                               f"Error message: {e}")
    return initialized_obj

  def resolve_serialize_refs(self, root):
#     for _, node in tree_tools.traverse_serializable_breadth_first(root):
    for _, node in tree_tools.traverse_serializable(root):
      if isinstance(node, Serializable):
        if not hasattr(node, "serialize_params"):
          raise ValueError(f"Cannot serialize node that has no serialize_params attribute: {node}\n"
                           "Did you forget to wrap the __init__() in @serializable_init ?")
        node.resolved_serialize_params = node.serialize_params
    refs_inserted_at = set()
    refs_inserted_to = set()
#     for path_to, node in tree_tools.traverse_serializable_breadth_first(root):
    for path_to, node in tree_tools.traverse_serializable(root):
      if not refs_inserted_at & path_to.ancestors() and not refs_inserted_at & path_to.ancestors():
        if isinstance(node, Serializable):
#           for path_from, matching_node in tree_tools.traverse_serializable_breadth_first(root):
          for path_from, matching_node in tree_tools.traverse_serializable(root):
            if not path_from in refs_inserted_to:
              if path_from!=path_to and matching_node is node:
                  ref = tree_tools.Ref(path=path_to)
                  ref.resolved_serialize_params = ref.serialize_params
                  tree_tools.set_descendant(root, path_from.parent().append("resolved_serialize_params").append(path_from[-1]), ref)
                  refs_inserted_at.add(path_from)
                  refs_inserted_to.add(path_from)

  def dump(self, ser_obj):
    self.resolve_serialize_refs(ser_obj)
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
      exp_global = UninitializedYamlObject(dict_spec)
    return corpus_parser, model, exp_global


class ComponentInitError(Exception):
  pass

yaml_serializer = YamlSerializer()

def serializable_init(f):
  @wraps(f)
  def wrapper(obj, *args, **kwargs):
    serialize_params = dict(kwargs)
    params = inspect.signature(f).parameters
    if len(args)>0:
      param_names = [p.name for p in list(params.values())]
      assert param_names[0] == "self"
      param_names = param_names[1:]
      for i, arg in enumerate(args):
        serialize_params[param_names[i]] = arg
    auto_added_defaults = set()
    for param in params.values():
      if param.name != "self" and param.default != inspect.Parameter.empty and param.name not in serialize_params:
        serialize_params[param.name] = param.default
        auto_added_defaults.add(param.name)
    for key, arg in serialize_params.items():
      if isinstance(arg, Ref):
        if not arg.is_required():
          serialize_params[key] = arg.get_default()
        else:
          if key in auto_added_defaults:
            raise ValueError(f"Required argument '{key}' of {type(obj).__name__}.__init__() was not specified, and {arg} could not be resolved")
          else:
            raise ValueError(f"Cannot pass a reference as argument; received {arg} in {type(obj).__name__}.__init__()")
      if getattr(arg, "_is_bare", False):
        serialize_params[key] = yaml_serializer.initialize_object(UninitializedYamlObject(arg))
    f(obj, **serialize_params)
    serialize_params.update(getattr(obj,"serialize_params",{}))
    obj.serialize_params = serialize_params
  return wrapper

