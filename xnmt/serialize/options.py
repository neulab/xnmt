"""
Stores options and default values
"""
import random
import inspect

import yaml

from xnmt.serialize.serializable import Serializable, UninitializedYamlObject
import xnmt.serialize.tree_tools as tree_tools
from xnmt.serialize.tree_tools import set_descendant

class Option(object):
  def __init__(self, name, opt_type=str, default_value=None, required=None, force_flag=False, help_str=None):
    """
    Defines a configuration option
    :param name: Name of the option
    :param opt_type: Expected type. Should be a base type.
    :param default_value: Default option value. If this is set to anything other than none, and the option is not
    explicitly marked as required, it will be considered optional.
    :param required: Whether the option is required.
    :param force_flag: Force making this argument a flag (starting with '--') even though it is required
    :param help_str: Help string for documentation
    """
    self.name = name
    self.type = opt_type
    self.default_value = default_value
    self.required = required == True or required is None and default_value is None
    self.force_flag = force_flag
    self.help = help_str

class RandomParam(yaml.YAMLObject):
  yaml_tag = u'!RandomParam'
  def __init__(self, values):
    self.values = values
  def __repr__(self):
    return f"{self.__class__.__name__}(values={self.values})"
  def draw_value(self):
    if not hasattr(self, 'drawn_value'):
      self.drawn_value = random.choice(self.values)
    return self.drawn_value

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

    if "defaults" in experiments: del experiments["defaults"]
    return sorted(experiments.keys())
    
  def parse_experiment(self, filename, exp_name):
    """
    Returns a dictionary of experiments => {task => {arguments object}}
    """
    try:
      with open(filename) as stream:
        config = yaml.load(stream)
    except IOError as e:
      raise RuntimeError(f"Could not read configuration file {filename}: {e}")

    experiment = config[exp_name]    

    for _, node in tree_tools.traverse_tree(experiment):
      if isinstance(node, Serializable):
        self.resolve_kwargs(node)

    for _, node in tree_tools.traverse_tree(experiment):
      if isinstance(node, Serializable):
        self.resolve_saved_model_file(node)
    
    random_search_report = self.instantiate_random_search(experiment)
    if random_search_report:
      experiment['random_search_report'] = random_search_report
    self.replace_placeholder(experiment, exp_name)

    return UninitializedYamlObject(experiment)

  def resolve_saved_model_file(self, obj):
    """
    Load the saved object and copy over attributes, unless they are overwritten in obj
    """
    if hasattr(obj, "pretrained_model_file"):
      try:
        with open(obj.pretrained_model_file) as stream:
          saved_obj = yaml.load(stream)
      except IOError as e:
        raise RuntimeError("Could not read configuration file {}: {}".format(obj.pretrained_model_file, e))
      saved_obj_items = inspect.getmembers(saved_obj)
      for name, _ in saved_obj_items:
        if not hasattr(obj, name):
          if name!="model_file":
            setattr(obj, name, getattr(saved_obj, name))
  
  def resolve_kwargs(self, obj):
    """
    If obj has a kwargs attribute (dictionary), set the dictionary items as attributes
    of the object via setattr (asserting that there are no collisions).
    """
    if hasattr(obj, "kwargs"):
      for k, v in obj.kwargs.items():
        if hasattr(obj, k):
          raise ValueError("kwargs %s already specified as class member for object %s" % (str(k), str(obj)))
        setattr(obj, k, v)
      delattr(obj, "kwargs")

  def instantiate_random_search(self, exp_values):
    param_report = {}
    initialized_random_params={}
    for path, v in tree_tools.traverse_tree(exp_values):
      if isinstance(v, RandomParam):
        if hasattr(v, "_xnmt_id") and v._xnmt_id in initialized_random_params:
          v = initialized_random_params[v._xnmt_id]
        v = v.draw_value()
        if hasattr(v, "_xnmt_id"):
          initialized_random_params[v._xnmt_id] = v
        set_descendant(exp_values, path, v)
        param_report[path] = v
    return param_report

  def replace_placeholder(self, exp_values, value, placeholder="<EXP>"):
    for path, node in tree_tools.traverse_tree(exp_values):
      if isinstance(node, str) and "<EXP>" in node:
        tree_tools.set_descendant(exp_values, path, node.replace(placeholder, value))
