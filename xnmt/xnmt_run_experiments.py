#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs to files called <experimentname>.log
and <experimentname>.err.log, and reporting on final perplexity metrics.
"""

import argparse
import sys
import six
import random
import shutil
import numpy as np

# XNMT imports
import xnmt.xnmt_preproc, xnmt.xnmt_evaluate, xnmt.training_regimen, xnmt.training_task, xnmt.eval_task
from xnmt.options import OptionParser
from xnmt.tee import Tee
from xnmt.serializer import YamlSerializer, UninitializedYamlObject
from xnmt.model_context import ModelContext, PersistentParamCollection

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
  argparser.add_argument("--generate-doc", action='store_true', help="Do not run, output documentation instead")
  argparser.add_argument("experiments_file")
  argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
  argparser.set_defaults(generate_doc=False)
  args = argparser.parse_args(overwrite_args)

  config_parser = OptionParser()

  if args.generate_doc:
    print(config_parser.generate_options_table())
    exit(0)

  if args.dynet_seed:
    random.seed(args.dynet_seed)
    np.random.seed(args.dynet_seed)

  config_experiment_names = config_parser.experiment_names_from_file(args.experiments_file)

  results = []

  # Check ahead of time that all experiments exist, to avoid bad surprises
  experiment_names = args.experiment_name or config_experiment_names

  if args.experiment_name:
    nonexistent = set(experiment_names).difference(config_experiment_names)
    if len(nonexistent) != 0:
      raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

  for experiment_name in experiment_names:
    exp_args = config_parser.parse_experiment(args.experiments_file, experiment_name)

    print("=> Running {}".format(experiment_name))

    yaml = YamlSerializer()

    glob_args = exp_args.get("global", {})
    out_file = glob_args.pop("out_file", "<EXP>.out")
    err_file = glob_args.pop("err_file", "<EXP>.err")

    output = Tee(out_file, 3)
    err_output = Tee(err_file, 3, error=True)

    model_file = glob_args.pop("model_file", "<EXP>.mod")
    cfg_file = glob_args.pop("cfg_file", None)
    save_num_checkpoints = glob_args.pop("save_num_checkpoints", 1)
    eval_only = glob_args.pop("eval_only", False)
    if cfg_file:
      shutil.copyfile(args.experiments_file, cfg_file)
    
    model_context = ModelContext()
    model_context.dynet_param_collection = PersistentParamCollection(model_file, save_num_checkpoints)
    for k, v in glob_args.items():
      setattr(model_context, k, v)

    print("> Preprocessing")
    preproc_args = exp_args.get("preproc", {})
    xnmt.xnmt_preproc.xnmt_preproc(**preproc_args)

    if "model" in exp_args:
      print("> Initializing Model")
      model_args = exp_args["model"]
      model_args = yaml.initialize_if_needed(UninitializedYamlObject(model_args), model_context)

    print("> Initializing TrainingRegimen")
    train_args = exp_args["train"]
    train_args.dynet_profiling = args.dynet_profiling
    train_args = yaml.initialize_if_needed(UninitializedYamlObject(train_args), model_context)

    # Do training
    if "random_search_report" in exp_args:
      print("> instantiated random parameter search: %s" % exp_args["random_search_report"])

    print("> Training")
    eval_scores = "Not evaluated"
    if not eval_only:
      train_args.run_training()
      print('reverting learned weights to best checkpoint..')
      train_args.yaml_context.dynet_param_collection.revert_to_best_model()

    evaluate_args = exp_args["evaluate"]
    if evaluate_args:
      print("> Performing final evaluation")
      output.indent += 2
      eval_scores = []
      for evaluator in evaluate_args:
        evaluator = yaml.initialize_if_needed(UninitializedYamlObject(evaluator), model_context)
        eval_score = evaluator.eval()
        if type(eval_score) == list:
          eval_scores.extend(eval_score)
        else:
          eval_scores.append(eval_score)
      output.indent -= 2

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
