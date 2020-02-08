#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file and runs them sequentially, logging outputs
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
import traceback
from typing import Optional, Sequence, Tuple
import shutil

import numpy as np

import xnmt
from xnmt.settings import settings
from xnmt import logger, file_logger, tee, utils
from xnmt.tee import log_preamble
from xnmt.param_collections import ParamManager
from xnmt.persistence import YamlPreloader, save_to_file, initialize_if_needed
from xnmt.eval import metrics

if xnmt.backend_torch:
  import torch

if settings.RESOURCE_WARNINGS:
  import warnings
  warnings.simplefilter('always', ResourceWarning)

def main(overwrite_args: Optional[Sequence[str]] = None) -> None:

  with tee.Tee(), tee.Tee(error=True):
    argparser = argparse.ArgumentParser()
    utils.add_backend_argparse(argparser)
    argparser.add_argument("--settings", type=str, default="standard", help="settings (standard, debug, or unittest)"
                                                                            "must be given in '=' syntax, e.g."
                                                                            " --settings=standard")
    argparser.add_argument("--resume", action='store_true', help="whether a saved experiment is being resumed, and"
                                                                 "locations of output files should be re-used.")
    argparser.add_argument("--backend", type=str, default="dynet", help="backend (dynet or torch)")
    argparser.add_argument("experiments_file")
    argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
    argparser.set_defaults(generate_doc=False)
    args = argparser.parse_args(overwrite_args)

    if xnmt.backend_dynet and args.dynet_seed: args.seed = args.dynet_seed
    if getattr(args, "seed", None):
      random.seed(args.seed)
      np.random.seed(args.seed)
      if xnmt.backend_torch: torch.manual_seed(0)

    if xnmt.backend_dynet and args.dynet_gpu and settings.CHECK_VALIDITY:
      settings.CHECK_VALIDITY = False
      log_preamble("disabling CHECK_VALIDITY because it is not supported in the DyNet/GPU setting", logging.WARNING)

    config_experiment_names = YamlPreloader.experiment_names_from_file(args.experiments_file)

    results = []

    # Check ahead of time that all experiments exist, to avoid bad surprises
    experiment_names = args.experiment_name or config_experiment_names

    if args.experiment_name:
      nonexistent = set(experiment_names).difference(config_experiment_names)
      if len(nonexistent) != 0:
        raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

    log_preamble(f"running XNMT revision {tee.get_git_revision()} on {socket.gethostname()} with {'DyNet' if xnmt.backend_dynet else 'PyTorch'} on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for experiment_name in experiment_names:

      ParamManager.init_param_col()

      uninitialized_exp_args = YamlPreloader.preload_experiment_from_file(args.experiments_file, experiment_name,
                                                                          resume=args.resume)

      logger.info(f"=> Running {experiment_name}")

      glob_args = uninitialized_exp_args.data.exp_global
      log_file = glob_args.log_file

      if not settings.OVERWRITE_LOG:
        log_files_exist = []
        if os.path.isfile(log_file): log_files_exist.append(log_file)
        if os.path.isdir(log_file + ".tb"): log_files_exist.append(log_file + ".tb/")
        if log_files_exist:
          logger.warning(f"log file(s) {' '.join(log_files_exist)} already exists, skipping experiment; "
                         f"please delete log file by hand if you want to overwrite it "
                         f"(or activate OVERWRITE_LOG, by either specifying an environment variable OVERWRITE_LOG=1, "
                         f"or specifying --settings=debug, or changing xnmt.settings.Standard.OVERWRITE_LOG manually)")
          continue
      elif settings.OVERWRITE_LOG and os.path.isdir(log_file + ".tb"):
        shutil.rmtree(log_file + ".tb/")  # remove tensorboard logs from previous run that is being overwritten

      tee.set_out_file(log_file, exp_name=experiment_name)

      try:

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

      except Exception as e:
        file_logger.error(traceback.format_exc())
        raise e
      finally:
        tee.unset_out_file()
    
def print_results(results: Sequence[Tuple[str,Sequence[metrics.EvalScore]]]):
  print("")
  print("{:<30}|{:<40}".format("Experiment", " Final Scores"))
  print("-" * (70 + 1))

  for experiment_name, eval_scores in results:
    for i in range(len(eval_scores)):
      print("{:<30}| {:<40}".format((experiment_name if i==0 else ""), str(eval_scores[i])))


if __name__ == '__main__':
  sys.exit(main())
