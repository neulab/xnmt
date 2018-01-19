#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs
"""

import argparse
import sys
import random

import numpy as np

from xnmt.serialize.options import OptionParser
from xnmt.tee import Tee
from xnmt.serialize.serializer import YamlSerializer

def main(overwrite_args=None):
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--dynet-mem", type=int)
  argparser.add_argument("--dynet-seed", type=int)
  argparser.add_argument("--dynet-autobatch", type=int)
  argparser.add_argument("--dynet-devices", type=str)
  argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
  argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
  argparser.add_argument("--dynet-gpu-ids", type=int)
  argparser.add_argument("--dynet-gpus", type=int)
  argparser.add_argument("--dynet-weight-decay", type=float)
  argparser.add_argument("--dynet-profiling", type=int)
  argparser.add_argument("experiments_file")
  argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
  argparser.set_defaults(generate_doc=False)
  args = argparser.parse_args(overwrite_args)

  config_parser = OptionParser()

  if args.dynet_seed:
    random.seed(args.dynet_seed)
    np.random.seed(args.dynet_seed)

  import xnmt.serialize.imports
  config_experiment_names = config_parser.experiment_names_from_file(args.experiments_file)

  results = []

  # Check ahead of time that all experiments exist, to avoid bad surprises
  experiment_names = args.experiment_name or config_experiment_names

  if args.experiment_name:
    nonexistent = set(experiment_names).difference(config_experiment_names)
    if len(nonexistent) != 0:
      raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

  for experiment_name in experiment_names:
    uninitialized_exp_args = config_parser.parse_experiment(args.experiments_file, experiment_name)

    print("=> Running {}".format(experiment_name))

    yaml_serializer = YamlSerializer()

    glob_args = uninitialized_exp_args.data.model_context
    out_file = glob_args.get_out_file(experiment_name)
    err_file = glob_args.get_err_file(experiment_name)

    output = Tee(out_file, 3)
    err_output = Tee(err_file, 3, error=True)

    model_file = glob_args.get_model_file(experiment_name)

    uninitialized_exp_args.data.model_context.commandline_args = args

    # TODO: Delete or move inside PreprocRunner.__init__()
#     if "preproc" in uninitialized_exp_args.data:
#       print("> Preprocessing")
#       preproc_args = uninitialized_exp_args.data.get("preproc", {})
#       del uninitialized_exp_args.data["preproc"]
#       yaml_serializer.initialize_if_needed(preproc_args, model_context)
#       xnmt.xnmt_preproc.xnmt_preproc(**preproc_args)

    # Create the model
    experiment = yaml_serializer.initialize_if_needed(uninitialized_exp_args)

    # Run the experiment
    eval_scores = experiment(save_fct = lambda: yaml_serializer.save_to_file(model_file, experiment,
                                                                             experiment.model_context.dynet_param_collection))
    results.append((experiment_name, eval_scores))

    output.close()
    err_output.close()

  print("")
  print("{:<30}|{:<40}".format("Experiment", " Final Scores"))
  print("-" * (70 + 1))

  for line in results:
    experiment_name, eval_scores = line
    for i in range(len(eval_scores)):
      print("{:<30}| {:<40}".format((experiment_name if i==0 else ""), str(eval_scores[i])))

if __name__ == '__main__':
  import _dynet
  dyparams = _dynet.DynetParams()
  dyparams.from_args()
  sys.exit(main())
