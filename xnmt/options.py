"""
Stores options and default values
"""
import yaml
import argparse
from collections import OrderedDict
import copy
import random
import inspect
from xnmt.serializer import Serializable, YamlSerializer

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
    return "%s(values=%r)" % (
            self.__class__.__name__, self.values)
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
      raise RuntimeError("Could not read configuration file {}: {}".format(filename, e))

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
      raise RuntimeError("Could not read configuration file {}: {}".format(filename, e))

    experiment = config[exp_name]    
    YamlSerializer.apply_to_serializable_recursive(experiment, self.resolve_kwargs)
    
    YamlSerializer.apply_to_serializable_recursive(experiment, self.resolve_saved_model_file)
    
    random_search_report = self.instantiate_random_search(experiment)
    if random_search_report:
      experiment['random_search_report'] = random_search_report
    self.replace_placeholder(experiment, exp_name)

    return experiment

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

  def instantiate_random_search(self, exp_values, initialized_random_params={}):
    param_report = {}
    if isinstance(exp_values, dict): kvs = exp_values.items()
    elif isinstance(exp_values, Serializable):
      init_args, _, _, _ = inspect.getargspec(exp_values.__init__)
      kvs = [(key, getattr(exp_values, key)) for key in init_args if hasattr(exp_values, key)]
    for k, v in kvs:
      if isinstance(v, RandomParam):
        if hasattr(v, "_xnmt_id") and v._xnmt_id in initialized_random_params:
          v = initialized_random_params[v._xnmt_id]
        v = v.draw_value()
        if hasattr(v, "_xnmt_id"):
          initialized_random_params[v._xnmt_id] = v
        if isinstance(exp_values, dict):
          exp_values[k] = v
        else:
          setattr(exp_values, k, v)
        param_report[k] = v
      elif isinstance(v, dict) or isinstance(v, Serializable):
        sub_report = self.instantiate_random_search(v, initialized_random_params)
        if sub_report:
          param_report[k] = sub_report
    return param_report

  def replace_placeholder(self, exp_values, value, placeholder="<EXP>"):
    if isinstance(exp_values, dict): kvs = exp_values.items()
    elif isinstance(exp_values, Serializable):
      init_args, _, _, _ = inspect.getargspec(exp_values.__init__)
      kvs = [(key, getattr(exp_values, key)) for key in init_args if hasattr(exp_values, key)]
    for k, v in kvs:
      if isinstance(v, str):
        if placeholder in v:
          if isinstance(exp_values, dict):
            exp_values[k] = v.replace(placeholder, value)
          else:
            setattr(exp_values, k, v.replace(placeholder, value))
      elif isinstance(v, dict) or isinstance(v, Serializable):
        self.replace_placeholder(v, value, placeholder)


  def args_from_command_line(self, task, argv):
    raise NotImplementedError("")
  # old code:
#    parser = argparse.ArgumentParser()
#    for option in self.tasks[task].values():
#      if option.required and not option.force_flag:
#        parser.add_argument(option.name, type=option.type, help=option.help)
#      else:
#        parser.add_argument("--" + option.name, default=option.default_value, required=option.required,
#                            type=option.type, help=option.help)
#
#    return parser.parse_args(argv)

  def generate_options_table(self):
    """
    Generates markdown documentation for the options
    """
    lines = []
    for task, task_options in self.tasks.items():
      lines.append("## {}".format(task))
      lines.append("")
      lines.append("| Name | Description | Type | Default value |")
      lines.append("|------|-------------|------|---------------|")
      for option in task_options.values():
        if option.required:
          template = "| **{}** | {} | {} | {} |"
        else:
          template = "| {} | {} | {} | {} |"
        lines.append(template.format(option.name, option.help if option.help else "", option.type.__name__,
                                     option.default_value if option.default_value is not None else ""))
      lines.append("")

    return "\n".join(lines)


# Predefined options for dynet
general_options = [
  Option("dynet_mem", int, required=False),
  Option("dynet_seed", int, required=False),
]
