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

    self.load_referenced_model(experiment)
    
    random_search_report = self.instantiate_random_search(experiment)
    if random_search_report:
      setattr(experiment, 'random_search_report', random_search_report)
    self.format_strings(experiment, {"EXP":exp_name })

    return UninitializedYamlObject(experiment)

  def load_referenced_model(self, experiment):
    if hasattr(experiment, "load") or hasattr(experiment, "overwrite"):
      exp_args = set([x[0] for x in tree_tools.name_children(experiment, include_reserved=False)])
      if exp_args not in [set(["load"]), set(["load","overwrite"])]:
        raise ValueError(f"When loading a model from an external YAML file, only 'load' and 'overwrite' are permitted ('load' is required) as arguments to the experiment. Found: {exp_args}")
      try:
        with open(experiment.load) as stream:
          saved_obj = yaml.load(stream)
      except IOError as e:
        raise RuntimeError(f"Could not read configuration file {experiment.load}: {e}")
      for saved_key, saved_val in tree_tools.name_children(saved_obj, include_reserved=True):
        if not hasattr(experiment, saved_key):
          setattr(experiment, saved_key, saved_val)

      if hasattr(experiment, "overwrite"):
        for d in experiment.overwrite:
          path = tree_tools.Path(d[0])
          try:
            tree_tools.set_descendant(experiment, path, d[1])
          except:
            tree_tools.set_descendant(experiment, path, d[1])
        delattr(experiment, "overwrite")
      
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

  def format_strings(self, exp_values, format_dict):
    for path, node in tree_tools.traverse_tree(exp_values):
      if isinstance(node, str):
        formatted = node.format(**format_dict)
        if node != formatted:
          tree_tools.set_descendant(exp_values,
                                    path,
                                    FormatString(formatted, node))
