"""
Stores options and default values for 
"""
import yaml
import argparse
from collections import OrderedDict
import copy
import random

class Option:
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


class Args: pass


class RandomParam(yaml.YAMLObject):
  yaml_tag = u'!RandomParam'
  def __init__(self, values):
    self.values = values
  def __repr__(self):
    return "%s(values=%r)" % (
            self.__class__.__name__, self.values)
  def draw_value(self):
    return random.choice(self.values)
  def apply_type(self, wanted_type):
    self.values = [wanted_type(v) for v in self.values]

class RefParam(yaml.YAMLObject):
  yaml_tag = u'!RefParam'
  def __init__(self, ref):
    self.ref = ref
  def __repr__(self):
    return "%s(ref=%r)" % (
            self.__class__.__name__, self.ref)
  def apply_type(self, wanted_type):
    self.wanted_type = wanted_type
  def get_typed(self, val):
    return self.wanted_type(val)

class OptionParser:
  def __init__(self):
    self.tasks = {}
    """Options, sorted by task"""

  def add_task(self, task_name, task_options):
    self.tasks[task_name] = OrderedDict([(opt.name, opt) for opt in task_options])

  def check_and_convert(self, task_name, option_name, value):
    if option_name not in self.tasks[task_name]:
      raise RuntimeError("Unknown option {} for task {}".format(option_name, task_name))

    option = self.tasks[task_name][option_name]
    if isinstance(value, RandomParam) or isinstance(value, RefParam):
      value.apply_type(option.type)
    else:
      value = option.type(value)

    return value
  
  def collapse_sub_dict(self, flat_dict):
    """
    {"encoder":{},"encoder.layers":1}
    => {"encoder":{"layers":1}}
    """
    ret = dict(flat_dict)
    for key in flat_dict.keys():
      if "." in key:
        key1, key2 = key.split(".", 1)
        if not key1 in ret:
          ret[key1] = {}
        ret[key1][key2] = flat_dict[key]
        del ret[key]
    for key, val in ret.items():
      if type(val)==dict:
        ret[key] = self.collapse_sub_dict(val)
    return ret
  def update_with_subdicts(self, dict_to_update, dict_new_vals):
    dict_to_update.update({key:value for key,value in dict_new_vals.items() if type(value)!=dict})
    for key,value in dict_new_vals.items():
      if type(value)==dict:
        if key not in dict_to_update:
          dict_to_update[key] = value
        else:
          self.update_with_subdicts(dict_to_update[key], value)

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
      task_name: self.collapse_sub_dict({name: opt.default_value for name, opt in task_options.items() if
                  opt.default_value is not None or not opt.required})
      for task_name, task_options in self.tasks.items()}

    # defaults section in the config file
    if "defaults" in config:
      for task_name, task_options in config["defaults"].items():
        self.update_with_subdicts(defaults[task_name],
                                  {name: self.check_and_convert(task_name, name, value) for name, value in task_options.items()})
      del config["defaults"]

    experiments = {}
    for exp, exp_tasks in config.items():
      if exp_tasks is None: exp_tasks = {}
      experiments[exp] = {}
      for task_name in self.tasks:
        task_values = copy.deepcopy(defaults[task_name])
        exp_task_values = exp_tasks.get(task_name, dict())
        self.update_with_subdicts(task_values,
                                  {name: self.check_and_convert(task_name, name, value) for name, value in exp_task_values.items()})

        # Check that no required option is missing
        for _, option in self.tasks[task_name].items():
          if option.required:
            sub_task_values = task_values
            sub_option_name = option.name
            while "." in sub_option_name:
              key1,key2 = sub_option_name.split(".",1)
              if key1 not in sub_task_values:
                raise RuntimeError(
                  "Required option not found for experiment {}, task {}: {}".format(exp, task_name, option.name))
              sub_task_values = sub_task_values[key1]
              sub_option_name = key2
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

    return experiments
  
  def instantiate_random_search(self, task_values):
    param_report = {}
    for k in task_values.keys():
      if isinstance(task_values[k], RandomParam):
        task_values[k] = task_values[k].draw_value()
        param_report[k] = task_values[k]
      elif type(task_values[k]) == dict:
        sub_report = self.instantiate_random_search(task_values[k])
        if sub_report:
          param_report[k] = sub_report
    return param_report
  def resolve_referenced_params(self, cur_task_values, top_task_values):
    for k in cur_task_values.keys():
      if isinstance(cur_task_values[k], RefParam):
        ref_str_spl = cur_task_values[k].ref.split(".")
        resolved = top_task_values
        for ref_str in ref_str_spl:
          resolved = resolved[ref_str]
        cur_task_values[k] = cur_task_values[k].get_typed(resolved)
      elif type(cur_task_values[k]) == dict:
        self.resolve_referenced_params(cur_task_values[k], top_task_values)
    

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
