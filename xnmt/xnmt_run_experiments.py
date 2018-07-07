#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs
"""
import argparse
import logging
import os
import random
import sys
import socket
import datetime
import faulthandler
faulthandler.enable()

import numpy as np
from xnmt.settings import settings

from xnmt import logger
from xnmt.tee import log_preamble
from xnmt.param_collection import ParamManager
import xnmt.tee as tee
from xnmt.persistence import YamlPreloader, save_to_file, initialize_if_needed

if settings.RESOURCE_WARNINGS:
  import warnings
  warnings.simplefilter('always', ResourceWarning)

def main(overwrite_args=None):

  with tee.Tee(), tee.Tee(error=True):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dynet-mem", type=str)
    argparser.add_argument("--dynet-seed", type=int, help="set random seed for DyNet and XNMT.")
    argparser.add_argument("--dynet-autobatch", type=int)
    argparser.add_argument("--dynet-devices", type=str)
    argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
    argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
    argparser.add_argument("--dynet-gpu-ids", type=int)
    argparser.add_argument("--dynet-gpus", type=int)
    argparser.add_argument("--dynet-weight-decay", type=float)
    argparser.add_argument("--dynet-profiling", type=int)
    argparser.add_argument("--settings", type=str, default="standard", help="settings (standard, debug, or unittest)"
                                                                            "must be given in '=' syntax, e.g."
                                                                            " --settings=standard")
    argparser.add_argument("experiments_file")
    argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
    argparser.set_defaults(generate_doc=False)
    args = argparser.parse_args(overwrite_args)

    if args.dynet_seed:
      random.seed(args.dynet_seed)
      np.random.seed(args.dynet_seed)

    if args.dynet_gpu:
      if settings.CHECK_VALIDITY:
        settings.CHECK_VALIDITY = False
        log_preamble("disabling CHECK_VALIDITY because it is not supported on GPU currently", logging.WARNING)

    config_experiment_names = YamlPreloader.experiment_names_from_file(args.experiments_file)

    results = []

    # Check ahead of time that all experiments exist, to avoid bad surprises
    experiment_names = args.experiment_name or config_experiment_names

    if args.experiment_name:
      nonexistent = set(experiment_names).difference(config_experiment_names)
      if len(nonexistent) != 0:
        raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

    log_preamble(f"running XNMT revision {tee.get_git_revision()} on {socket.gethostname()} on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for experiment_name in experiment_names:

      ParamManager.init_param_col()

      uninitialized_exp_args = YamlPreloader.preload_experiment_from_file(args.experiments_file, experiment_name)

      logger.info(f"=> Running {experiment_name}")

      glob_args = uninitialized_exp_args.data.exp_global
      log_file = glob_args.log_file

      if os.path.isfile(log_file) and not settings.OVERWRITE_LOG:
        logger.warning(f"log file {log_file} already exists, skipping experiment; please delete log file by hand if you want to overwrite it "
                       f"(or activate OVERWRITE_LOG, by either specifying an environment variable as OVERWRITE_LOG=1, "
                       f"or specifying --settings=debug, or changing xnmt.settings.Standard.OVERWRITE_LOG manually)")
        continue

      tee.set_out_file(log_file)

      model_file = glob_args.model_file

      uninitialized_exp_args.data.exp_global.commandline_args = vars(args)

      # Create the model
      experiment = initialize_if_needed(uninitialized_exp_args)
      ParamManager.param_col.model_file = experiment.exp_global.model_file
      ParamManager.param_col.save_num_checkpoints = experiment.exp_global.save_num_checkpoints
      ParamManager.populate()

      # Run the experiment
      eval_scores = experiment(save_fct = lambda: save_to_file(model_file, experiment))
      results.append((experiment_name, eval_scores))
      print_results(results)

      tee.unset_out_file()
    
def print_results(results):
  print("")
  print("{:<30}|{:<40}".format("Experiment", " Final Scores"))
  print("-" * (70 + 1))

  for experiment_name, eval_scores in results:
    for i in range(len(eval_scores)):
      print("{:<30}| {:<40}".format((experiment_name if i==0 else ""), str(eval_scores[i])))


if __name__ == '__main__':
  sys.exit(main())
