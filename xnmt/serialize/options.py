"""
Stores options and default values
"""
import logging
logger = logging.getLogger('xnmt')
import random
import os
import copy

import yaml

from xnmt.serialize.serializable import Serializable, UninitializedYamlObject
import xnmt.serialize.tree_tools as tree_tools
from xnmt.serialize.tree_tools import set_descendant
from xnmt.param_collection import ParamManager

class Option(object):
  def __init__(self, name, opt_type=str, default_value=None, required=None, force_flag=False, help_str=None):
    """
    Defines a configuration option

    Args:
      name: Name of the option
      opt_type: Expected type. Should be a base type.
      default_value: Default option value. If this is set to anything other than none, and the option is not
        explicitly marked as required, it will be considered optional.
      required: Whether the option is required.
      force_flag: Force making this argument a flag (starting with '--') even though it is required
      help_str: Help string for documentation
    """
    self.name = name
    self.type = opt_type
    self.default_value = default_value
    self.required = required == True or required is None and default_value is None
    self.force_flag = force_flag
    self.help = help_str

class FormatString(str, yaml.YAMLObject):
  """
  Used to handle the {EXP} string formatting syntax.
  When passed around it will appear like the properly resolved string,
  but writing it back to YAML will use original version containing {EXP}
  """
  def __new__(cls, value, *args, **kwargs):
    return super().__new__(cls, value)
  def __init__(self, value, serialize_as):
    self.serialize_as = serialize_as
def init_fs_representer(dumper, obj):
    return dumper.represent_data(obj.serialize_as)
yaml.add_representer(FormatString, init_fs_representer)

class RandomParam(yaml.YAMLObject):
  yaml_tag = '!RandomParam'
  def __init__(self, values):
    self.values = values
  def __repr__(self):
    return f"{self.__class__.__name__}(values={self.values})"
  def draw_value(self):
    if not hasattr(self, 'drawn_value'):
      self.drawn_value = random.choice(self.values)
    return self.drawn_value

class LoadSerialized(Serializable):
  yaml_tag = "!LoadSerialized"
  def __init__(self, filename, path="", overwrite=[]):
    self.filename = filename
    self.path = path
    self.overwrite = overwrite

class OptionParser(object):
  def __init__(self):
    self.tasks = {}
    """Options, sorted by task"""

  def experiment_names_from_file(self, filename):
    try:
      with open(filename) as stream:
        experiments = yaml.load(stream)
    except IOError as e:
      raise RuntimeError(f"Could not read configuration file {filename}: {e}")
    except yaml.constructor.ConstructorError:
      logger.error("for proper deserialization of a class object, make sure the class is a subclass of xnmt.serialize.serializable.Serializable, specifies a proper yaml_tag with leading '!', and its module is imported under xnmt/serialize/imports.py")
      raise

    if "defaults" in experiments: del experiments["defaults"]
    return sorted(experiments.keys())

  def parse_experiment_file(self, filename, exp_name):
    try:
      with open(filename) as stream:
        config = yaml.load(stream)
    except IOError as e:
      raise RuntimeError(f"Could not read configuration file {filename}: {e}")

    experiment = config[exp_name]
    return self.parse_loaded_experiment(experiment, exp_name = exp_name, exp_dir = os.path.dirname(filename))

  def parse_loaded_experiment(self, experiment, exp_name, exp_dir):

    for _, node in tree_tools.traverse_tree(experiment):
      if isinstance(node, Serializable):
        self.resolve_kwargs(node)

    experiment = self.load_referenced_serialized(experiment)

    random_search_report = self.instantiate_random_search(experiment)
    if random_search_report:
      setattr(experiment, 'random_search_report', random_search_report)

    # if arguments were not given in the YAML file and are set to a bare(Serializable) by default, copy the bare object into the object hierarchy so it can be used w/ param sharing etc.
    OptionParser.resolve_bare_default_args(experiment)
      
    self.format_strings(experiment, {"EXP":exp_name,"PID":os.getpid(),
                                     "EXP_DIR":exp_dir})

    return UninitializedYamlObject(experiment)

  def load_referenced_serialized(self, experiment):
    for path, node in tree_tools.traverse_tree(experiment):
      if isinstance(node, LoadSerialized):
        try:
          with open(node.filename) as stream:
            loaded_obj = yaml.load(stream)
        except IOError as e:
          raise RuntimeError(f"Could not read configuration file {node.filename}: {e}")
        ParamManager.add_load_path(f"{node.filename}.data")
        loaded_obj = tree_tools.get_descendant(loaded_obj, tree_tools.Path(getattr(node, "path", "")))
        # TODO: in case the components contain references to a component outside
        # the loaded path, we need to move it over and change other components'
        # references to the same object accordingly

        for d in getattr(node, "overwrite", []):
          overwrite_path = tree_tools.Path(d["path"])
          tree_tools.set_descendant(loaded_obj, overwrite_path, d["val"])
        if len(path)==0:
          experiment = loaded_obj
        else:
          set_descendant(experiment, path, loaded_obj)
    return experiment

  def resolve_kwargs(self, obj):
    """
    If obj has a kwargs attribute (dictionary), set the dictionary items as attributes
    of the object via setattr (asserting that there are no collisions).
    """
    if hasattr(obj, "kwargs"):
      for k, v in obj.kwargs.items():
        if hasattr(obj, k):
          raise ValueError(f"kwargs '{str(k)}' already specified as class member for object '{str(obj)}'")
        setattr(obj, k, v)
      delattr(obj, "kwargs")

  def instantiate_random_search(self, experiment):
    param_report = {}
    initialized_random_params={}
    for path, v in tree_tools.traverse_tree(experiment):
      if isinstance(v, RandomParam):
        if hasattr(v, "_xnmt_id") and v._xnmt_id in initialized_random_params:
          v = initialized_random_params[v._xnmt_id]
        v = v.draw_value()
        if hasattr(v, "_xnmt_id"):
          initialized_random_params[v._xnmt_id] = v
        set_descendant(experiment, path, v)
        param_report[path] = v
    return param_report

  @staticmethod
  def resolve_bare_default_args(root):
    for path, node in tree_tools.traverse_tree(root):
      if isinstance(node, Serializable):
        init_args_defaults = tree_tools.get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in tree_tools.name_children(node, include_reserved=False)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, Serializable) and not isinstance(arg_default, tree_tools.Ref):
              if not getattr(arg_default, "_is_bare", False):
                raise ValueError(f"only Serializables created via bare(SerializableSubtype) are permitted as default arguments; "
                                 f"found a fully initialized Serializable: {arg_default} at {path}")
              OptionParser.resolve_bare_default_args(arg_default) # apply recursively
              setattr(node, expected_arg, copy.deepcopy(arg_default))

  def format_strings(self, exp_values, format_dict):
    """
    - replaces strings containing {EXP} and other supported args
    - also checks if there are default arguments for which no arguments are set and instantiates them with replaced {EXP} if applicable
    """
    for path, node in tree_tools.traverse_tree(exp_values):
      if isinstance(node, str):
        try:
          formatted = node.format(**format_dict)
        except (ValueError, KeyError): # will occur e.g. if a vocab entry contains a curly bracket
          formatted = node
        if node != formatted:
          tree_tools.set_descendant(exp_values,
                                    path,
                                    FormatString(formatted, node))
      elif isinstance(node, Serializable):
        init_args_defaults = tree_tools.get_init_args_defaults(node)
        for expected_arg in init_args_defaults:
          if not expected_arg in [x[0] for x in tree_tools.name_children(node, include_reserved=False)]:
            arg_default = init_args_defaults[expected_arg].default
            if isinstance(arg_default, str):
              try:
                formatted = arg_default.format(**format_dict)
              except (ValueError, KeyError): # will occur e.g. if a vocab entry contains a curly bracket
                formatted = arg_default
              if arg_default != formatted:
                setattr(node, expected_arg, FormatString(formatted, arg_default))
        
