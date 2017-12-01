#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs to files called <experimentname>.log
and <experimentname>.err.log, and reporting on final perplexity metrics.
"""

import argparse
import sys
import os
import six
import random
import shutil
import numpy as np
import copy

# XNMT imports
import xnmt.xnmt_preproc, xnmt.xnmt_decode, xnmt.xnmt_evaluate, xnmt.train
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
    exp_tasks = config_parser.parse_experiment(args.experiments_file, experiment_name)

    print("=> Running {}".format(experiment_name))
    
    exp_args = exp_tasks.get("experiment", {})
    # TODO: refactor
    if not "model_file" in exp_args: exp_args["model_file"] = "<EXP>.mod"
    if not "hyp_file" in exp_args: exp_args["hyp_file"] = "<EXP>.hyp"
    if not "out_file" in exp_args: exp_args["out_file"] = "<EXP>.out"
    if not "err_file" in exp_args: exp_args["model_file"] = "<EXP>.err"
    if not "cfg_file" in exp_args: exp_args["cfg_file"] = None
    if not "eval_only" in exp_args: exp_args["eval_only"] = False
    if not "eval_metrics" in exp_args: exp_args["eval_metrics"] = "bleu"
    if not "save_num_checkpoints" in exp_args: exp_args["save_num_checkpoints"] = 1
    if "cfg_file" in exp_args and exp_args["cfg_file"] != None:
      shutil.copyfile(args.experiments_file, exp_args["cfg_file"])

    preproc_args = exp_tasks.get("preproc", {})
    # Do preprocessing
    print("> Preprocessing")
    xnmt.xnmt_preproc.xnmt_preproc(**preproc_args)

    print("> Initializing TrainingRegimen")
    train_args = exp_tasks["train"]
    train_args.model_file = exp_args["model_file"] # TODO: can we use param sharing for this?
    train_args.dynet_profiling = args.dynet_profiling
    model_context = ModelContext()
    model_context.dynet_param_collection = PersistentParamCollection(exp_args["model_file"], exp_args["save_num_checkpoints"])
    if hasattr(train_args, "glob"):
      for k in train_args.glob:
        setattr(model_context, k, train_args.glob[k])
    train_args = YamlSerializer().initialize_if_needed(UninitializedYamlObject(train_args), model_context)
    
    decode_args = exp_tasks.get("decode", {})
    decode_args["trg_file"] = exp_args["hyp_file"]
    decode_args["model_file"] = None  # The model is passed to the decoder directly

    evaluate_args = exp_tasks.get("evaluate", {})
    evaluate_args["hyp_file"] = exp_args["hyp_file"]
    evaluators = map(lambda s: s.lower(), exp_args["eval_metrics"].split(","))

    output = Tee(exp_args["out_file"], 3)
    err_output = Tee(exp_args["err_file"], 3, error=True)

    # Do training
    if "random_search_report" in exp_tasks:
      print("> instantiated random parameter search: %s" % exp_tasks["random_search_report"])

    print("> Training")
    training_regimen = train_args
    training_regimen.decode_args = copy.copy(decode_args)
    training_regimen.evaluate_args = copy.copy(evaluate_args)

    eval_scores = "Not evaluated"
    if not exp_args["eval_only"]:
      training_regimen.run_epochs()

    if not exp_args["eval_only"]:
      print('reverting learned weights to best checkpoint..')
      training_regimen.yaml_context.dynet_param_collection.revert_to_best_model()
    if evaluators:
      print("> Evaluating test set")
      output.indent += 2
      xnmt.xnmt_decode.xnmt_decode(model_elements=(
        training_regimen.corpus_parser, training_regimen.model), **decode_args)
      eval_scores = []
      for evaluator in evaluators:
        evaluate_args["evaluator"] = evaluator
        eval_score = xnmt.xnmt_evaluate.xnmt_evaluate(**evaluate_args)
        print(eval_score)
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
