#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs
"""
import logging
logger = logging.getLogger('xnmt')
import argparse
import os
import random
import sys
import socket
import faulthandler
faulthandler.enable()

import numpy as np
from simple_settings import settings
if settings.RESOURCE_WARNINGS:
  import warnings
  warnings.simplefilter('always', ResourceWarning)

from xnmt.serialize.options import OptionParser
from xnmt.tee import Tee, get_git_revision
from xnmt.serialize.serializer import YamlSerializer

def main(overwrite_args=None):

  with Tee(), Tee(error=True):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dynet-mem", type=str)
    argparser.add_argument("--dynet-seed", type=int)
    argparser.add_argument("--dynet-autobatch", type=int)
    argparser.add_argument("--dynet-devices", type=str)
    argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
    argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
    argparser.add_argument("--dynet-gpu-ids", type=int)
    argparser.add_argument("--dynet-gpus", type=int)
    argparser.add_argument("--dynet-weight-decay", type=float)
    argparser.add_argument("--dynet-profiling", type=int)
    argparser.add_argument("--settings", type=str, default="standard")
    argparser.add_argument("experiments_file")
    argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
    argparser.set_defaults(generate_doc=False)
    args = argparser.parse_args(overwrite_args)
  
    config_parser = OptionParser()
  
    if args.dynet_seed:
      random.seed(args.dynet_seed)
      np.random.seed(args.dynet_seed)

    if args.dynet_gpu:
      if settings.CHECK_VALIDITY:
        settings.CHECK_VALIDITY = False
        logger.warning("disabling CHECK_VALIDITY because it is not supported on GPU currently")
  
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
  
      logger.info("=> Running {}".format(experiment_name))
      logger.debug(f"running XNMT revision {get_git_revision()} on {socket.gethostname()}")

      yaml_serializer = YamlSerializer()
  
      glob_args = uninitialized_exp_args.data.exp_global
      log_file = glob_args.log_file
      
      if os.path.isfile(log_file) and not settings.OVERWRITE_LOG:
        logger.warning(f"log file {log_file} already exists; please delete by hand if you want to overwrite it (or use --settings=settings.debug or otherwise set OVERWRITE_LOG=True); skipping experiment..")
        continue
  
      xnmt.tee.set_out_file(log_file)
  
      model_file = glob_args.model_file
  
      uninitialized_exp_args.data.exp_global.commandline_args = args
  
      # Create the model
      experiment = yaml_serializer.initialize_if_needed(uninitialized_exp_args)
  
      # Run the experiment
      eval_scores = experiment(save_fct = lambda: yaml_serializer.save_to_file(model_file, experiment,
                                                                               experiment.exp_global.dynet_param_collection))
      results.append((experiment_name, eval_scores))
      print_results(results)
      
      xnmt.tee.unset_out_file()
    
def print_results(results):
  print("")
  print("{:<30}|{:<40}".format("Experiment", " Final Scores"))
  print("-" * (70 + 1))

  for experiment_name, eval_scores in results:
    for i in range(len(eval_scores)):
      print("{:<30}| {:<40}".format((experiment_name if i==0 else ""), str(eval_scores[i])))
  

if __name__ == '__main__':
  import _dynet
  dyparams = _dynet.DynetParams()
  dyparams.from_args()
  sys.exit(main())
