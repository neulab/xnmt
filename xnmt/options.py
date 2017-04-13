"""
Stores options and default values for 
"""
import yaml
import argparse


class Option:
  def __init__(self, name, type=str, default_value=None, required=None, force_flag=False):
    """
    Defines a configuration option
    :param name: Name of the option
    :param type: Expected type. Should be a base type.
    :param default_value: Default option value. If this is set to anything other than none, and the option is not
    explicitly marked as required, it will be considered optional.
    :param required: Whether the option is required
    :param force_flag: If set to true, the command-line version of this option will be made a flag option (starting
    with '--')
    """
    self.name = name
    self.type = type
    self.default_value = default_value
    self.required = required == True or required is None and default_value is None
    self.force_flag = force_flag


class Args: pass


class OptionParser:
  def __init__(self):
    self.tasks = {}
    """Options, sorted by task"""

  def add_task(self, task_name, task_options):
    self.tasks[task_name] = {opt.name: opt for opt in task_options}

  def check_and_convert(self, task_name, option_name, value):
    if option_name not in self.tasks[task_name]:
      raise RuntimeError("Unknown option {} for task {}".format(option_name, task_name))

    option = self.tasks[task_name][option_name]
    value = option.type(value)

    return value

  def args_from_config_file(self, file):
    """
    Returns a dictionary of experiments => {task => {arguments object}}
    """
    try:
      with open(file) as stream:
        config = yaml.load(stream)
    except IOError as e:
      raise RuntimeError("Could not read configuration file {}: {}".format(file, e))

    # Default values as specified in option definitions
    defaults = {task_name: {name: opt.default_value for name, opt in task_options.items() if opt.default_value is not None}
                for task_name, task_options in self.tasks.items()}

    # defaults section in the config file
    if "defaults" in config:
      for task_name, task_options in config["defaults"].items():
        defaults[task_name].update(
          {name: self.check_and_convert(task_name, name, value) for name, value in task_options.items()})
    del config["defaults"]

    print "Defaults:"
    print defaults

    experiments = {}
    for exp, exp_tasks in config.items():
      experiments[exp] = {}
      for task_name in self.tasks:
        task_values = defaults[task_name].copy()
        exp_task_values = exp_tasks.get(task_name, dict())
        task_values.update(
          {name: self.check_and_convert(task_name, name, value) for name, value in exp_task_values.items()})

        # Check that no required option is missing
        for _, option in self.tasks[task_name].items():
          if option.required and option.name not in task_values:
            raise RuntimeError(
              "Required option not found for experiment {}, task {}: {}".format(exp, task_name, option.name))

        experiments[exp][task_name] = Args()
        for name, val in task_values.items():
          setattr(experiments[exp][task_name], name, val)

    return experiments

  def args_from_command_line(self, task, argv):
    parser = argparse.ArgumentParser()
    for option in self.tasks[task].values():
      name = ("--" if option.force_flag else "") + option.name
      parser.add_argument(name, default=option.default_value, required=option.required, type=option.type)

    return parser.parse_args(argv)


# general_options = [
#   Option("dynet_mem", int, required=False, commandline_only=True, force_flag=True),
#   Option("dynet_seed", int, required=False, commandline_only=True, force_flag=True),
# ]
option_parser = OptionParser()

train_options = [
  Option("eval_every", int, default_value=1000, force_flag=True),
  Option("batch_size", int, default_value=32, force_flag=True),
  Option("batch_strategy", default_value="src"),
  Option("train_source"),
  Option("train_target"),
  Option("dev_source"),
  Option("dev_target"),
  Option("model_file"),
  Option("input_type", default_value="word"),
  Option("input_word_embed_dim", int, default_value=67),
  Option("output_word_embed_dim", int, default_value=67),
  Option("output_state_dim", int, default_value=67),
  Option("attender_hidden_dim", int, default_value=67),
  Option("output_mlp_hidden_dim", int, default_value=67),
  Option("encoder_hidden_dim", int, default_value=64),
  Option("trainer", default_value="sgd"),
  Option("eval_metrics", default_value="bleu"),
  Option("encoder_layers", int, default_value=2),
  Option("decoder_layers", int, default_value=2),
  Option("encoder_type", default_value="BiLSTM"),
  Option("decoder_type", default_value="LSTM"),
  Option("decode_every", int, default_value=0),
  Option("run_for_epochs", int, default_value=0),
]
option_parser.add_task("train", train_options)
decode_options = [
  Option("model_file"),
  Option("test_source"),
  Option("hyp_file"),
]
option_parser.add_task("decode", decode_options)
evaluate_options = [
  Option("hyp_file"),
  Option("test_target"),
  Option("evaluator", default_value="bleu")
]
option_parser.add_task("evaluate", evaluate_options)

f = option_parser.args_from_config_file("../test/new-config.yaml")

print "Final:"
print f
print type(f["experiment1"]["train"].encoder_type)
