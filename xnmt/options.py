"""
Stores options and default values
"""
import yaml
import argparse
from collections import OrderedDict
import copy
import random
import inspect
from xnmt.serializer import Serializable

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

class Args(object):
  def __init__(self, **kwargs):
    for key,val in kwargs.items():
      setattr(self, key, val)

class RandomParam(yaml.YAMLObject):
  yaml_tag = u'!RandomParam'
  def __init__(self, values):
    self.values = values
  def __repr__(self):
    return "%s(values=%r)" % (
            self.__class__.__name__, self.values)
  def draw_value(self):
    return random.choice(self.values)

class RefParam(yaml.YAMLObject):
  yaml_tag = u'!RefParam'
  def __init__(self, ref):
    self.ref = ref
  def __repr__(self):
    return "%s(ref=%r)" % (
            self.__class__.__name__, self.ref)

class OptionParser(object):
  def __init__(self):
    self.tasks = {}
    """Options, sorted by task"""

  def add_task(self, task_name, task_options):
    self.tasks[task_name] = OrderedDict([(opt.name, opt) for opt in task_options])

  def check_and_convert(self, task_name, option_name, value):
    if option_name not in self.tasks[task_name]:
      raise RuntimeError("Unknown option {} for task {}".format(option_name, task_name))

    option = self.tasks[task_name][option_name]
    if not (isinstance(value, RandomParam) or isinstance(value, RefParam) or isinstance(value, Serializable)):
      value = option.type(value)

    return value

  def args_from_config_file(self, filename):
    """
    Returns a dictionary of experiments => {task => {arguments object}}
    """
    try:
      with open(filename) as stream:
        config = yaml.load(stream)
    except IOError as e:
      raise RuntimeError("Could not read configuration file {}: {}".format(filename, e))

    # Default values as specified in option definitions
    defaults = {
      task_name: dict({name: opt.default_value for name, opt in task_options.items() if
                  opt.default_value is not None or not opt.required})
      for task_name, task_options in self.tasks.items()}

    # defaults section in the config file
    if "defaults" in config:
      for task_name, task_options in config["defaults"].items():
        defaults[task_name].update({name: self.check_and_convert(task_name, name, value) for name, value in task_options.items()})
      del config["defaults"]

    experiments = {}
    for exp, exp_tasks in config.items():
      if exp_tasks is None: exp_tasks = {}
      experiments[exp] = {}
      for task_name in self.tasks:
        task_values = copy.deepcopy(defaults[task_name])
        exp_task_values = exp_tasks.get(task_name, dict())
        task_values.update({name: self.check_and_convert(task_name, name, value) for name, value in exp_task_values.items()})

        # Check that no required option is missing
        for _, option in self.tasks[task_name].items():
          if option.required:
            sub_task_values = task_values
            sub_option_name = option.name
            if sub_option_name not in sub_task_values:
              raise RuntimeError(
                "Required option not found for experiment {}, task {}: {}".format(exp, task_name, option.name))

        # Replace the special token "<EXP>" with the experiment name if necessary
        for k in task_values.keys():
          if type(task_values[k]) == str:
            task_values[k] = task_values[k].replace("<EXP>", exp)

        random_search_report = self.instantiate_random_search(task_values)
        if random_search_report:
          task_values["random_search_report"] = random_search_report

        self.resolve_referenced_params(task_values, task_values)

        experiments[exp][task_name] = Args()
        for name, val in task_values.items():
          setattr(experiments[exp][task_name], name, val)
        setattr(experiments[exp][task_name], "params_as_dict", task_values)

    return experiments

  def instantiate_random_search(self, task_values):
    param_report = {}
    if isinstance(task_values, dict): kvs = task_values.items()
    elif isinstance(task_values, Serializable):
      init_args, _, _, _ = inspect.getargspec(task_values.__init__)
      kvs = [(key, getattr(task_values, key)) for key in init_args if hasattr(task_values, key)]
    for k, v in kvs:
      if isinstance(v, RandomParam):
        v = v.draw_value()
        if isinstance(task_values, dict):
          task_values[k] = v
        else:
          setattr(task_values, k, v)
        param_report[k] = v
      elif isinstance(v, dict) or isinstance(v, Serializable):
        sub_report = self.instantiate_random_search(v)
        if sub_report:
          param_report[k] = sub_report
    return param_report

  def resolve_referenced_params(self, cur_task_values, top_task_values):
    if isinstance(cur_task_values, dict): kvs = cur_task_values.items()
    elif isinstance(cur_task_values, Serializable):
      init_args, _, _, _ = inspect.getargspec(cur_task_values.__init__)
      kvs = [(key, getattr(cur_task_values, key)) for key in init_args if hasattr(cur_task_values, key)]
    else:
      raise RuntimeError()
    for k, v in kvs:
      if isinstance(v, RefParam):
        ref_str_spl = v.ref.split(".")
        resolved = top_task_values
        for ref_str in ref_str_spl:
          if isinstance(resolved, dict):
            resolved = resolved[ref_str]
          else:
            resolved = getattr(resolved, ref_str)
        if isinstance(cur_task_values, dict):
          cur_task_values[k] = resolved
        elif isinstance(cur_task_values, Serializable):
          setattr(cur_task_values, k, resolved)
      elif isinstance(v, dict) or isinstance(v, Serializable):
        self.resolve_referenced_params(v, top_task_values)


  def args_from_command_line(self, task, argv):
    parser = argparse.ArgumentParser()
    for option in self.tasks[task].values():
      if option.required and not option.force_flag:
        parser.add_argument(option.name, type=option.type, help=option.help)
      else:
        parser.add_argument("--" + option.name, default=option.default_value, required=option.required,
                            type=option.type, help=option.help)

    return parser.parse_args(argv)

  def remove_option(self, task, option_name):
    if option_name not in self.tasks[task]:
      raise RuntimeError("Tried to remove nonexistent option {} for task {}".format(option_name, task))
    del self.tasks[task][option_name]

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
