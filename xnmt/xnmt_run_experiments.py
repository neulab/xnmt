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

# XNMT imports
import copy
import xnmt.xnmt_preproc, xnmt.xnmt_train, xnmt.xnmt_decode, xnmt.xnmt_evaluate
from xnmt.options import OptionParser, Option
from xnmt.tee import Tee

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
  argparser.add_argument("--generate-doc", action='store_true', help="Do not run, output documentation instead")
  argparser.add_argument("experiments_file")
  argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
  argparser.set_defaults(generate_doc=False)
  args = argparser.parse_args(overwrite_args)

  config_parser = OptionParser()
  config_parser.add_task("preproc", xnmt.xnmt_preproc.options)
  config_parser.add_task("train", xnmt.xnmt_train.options)
  config_parser.add_task("decode", xnmt.xnmt_decode.options)
  config_parser.add_task("evaluate", xnmt.xnmt_evaluate.options)

  # Tweak the options to make config files less repetitive:
  # - Delete evaluate:evaluator, replace with exp:eval_metrics
  # - Delete decode:hyp_file, evaluate:hyp_file, replace with exp:hyp_file
  # - Delete train:model, decode:model_file, replace with exp:model_file
  config_parser.remove_option("evaluate", "evaluator")
  config_parser.remove_option("decode", "trg_file")
  config_parser.remove_option("evaluate", "hyp_file")
  config_parser.remove_option("train", "model_file")
  config_parser.remove_option("decode", "model_file")

  experiment_options = [
    Option("model_file", default_value="<EXP>.mod", help_str="Location to write the model file"),
    Option("hyp_file", default_value="<EXP>.hyp", help_str="Location to write decoded output for evaluation"),
    Option("out_file", default_value="<EXP>.out", help_str="Location to write stdout messages"),
    Option("err_file", default_value="<EXP>.err", help_str="Location to write stderr messages"),
    Option("cfg_file", default_value=None, help_str="Location to write a copy of the YAML configuration file", required=False),
    Option("eval_only", bool, default_value=False, help_str="Skip training and evaluate only"),
    Option("eval_metrics", default_value="bleu", help_str="Comma-separated list of evaluation metrics (bleu/wer/cer)"),
    Option("run_for_epochs", int, help_str="How many epochs to run each test for"),
  ]

  config_parser.add_task("experiment", experiment_options)

  if args.generate_doc:
    print(config_parser.generate_options_table())
    exit(0)

  if args.dynet_seed:
    random.seed(args.dynet_seed)
    np.random.seed(args.dynet_seed)

  config = config_parser.args_from_config_file(args.experiments_file)

  results = []

  # Check ahead of time that all experiments exist, to avoid bad surprises
  experiment_names = args.experiment_name or config.keys()

  if args.experiment_name:
    nonexistent = set(experiment_names).difference(config.keys())
    if len(nonexistent) != 0:
      raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

  for experiment_name in sorted(experiment_names):
    exp_tasks = config[experiment_name]

    print("=> Running {}".format(experiment_name))

    exp_args = exp_tasks["experiment"]
    if exp_args.cfg_file != None:
      shutil.copyfile(args.experiments_file, exp_args.cfg_file)

    preproc_args = exp_tasks["preproc"]

    train_args = exp_tasks["train"]
    train_args.model_file = exp_args.model_file

    decode_args = exp_tasks["decode"]
    decode_args.trg_file = exp_args.hyp_file
    decode_args.model_file = None  # The model is passed to the decoder directly

    evaluate_args = exp_tasks["evaluate"]
    evaluate_args.hyp_file = exp_args.hyp_file
    evaluators = map(lambda s: s.lower(), exp_args.eval_metrics.split(","))

    output = Tee(exp_args.out_file, 3)
    err_output = Tee(exp_args.err_file, 3, error=True)

    # Do preprocessing
    print("> Preprocessing")
    xnmt.xnmt_preproc.xnmt_preproc(preproc_args)

    # Do training
    for task_name in exp_tasks:
      if hasattr(exp_tasks[task_name], "random_search_report"):
        print("> instantiated random parameter search: %s" % exp_tasks[task_name].random_search_report)

    print("> Training")
    xnmt_trainer = xnmt.xnmt_train.XnmtTrainer(train_args)
    xnmt_trainer.decode_args = copy.copy(decode_args)
    xnmt_trainer.evaluate_args = copy.copy(evaluate_args)

    eval_scores = "Not evaluated"
    for i_epoch in six.moves.range(exp_args.run_for_epochs):
      if not exp_args.eval_only:
        xnmt_trainer.run_epoch()

      if xnmt_trainer.early_stopping_reached:
        break

    if not exp_args.eval_only:
      print('reverting learned weights to best checkpoint..')
      xnmt_trainer.model_context.dynet_param_collection.revert_to_best_model()
    if evaluators:
      print("> Evaluating test set")
      output.indent += 2
      xnmt.xnmt_decode.xnmt_decode(decode_args, model_elements=(
        xnmt_trainer.corpus_parser, xnmt_trainer.model))
      eval_scores = []
      for evaluator in evaluators:
        evaluate_args.evaluator = evaluator
        eval_score = xnmt.xnmt_evaluate.xnmt_evaluate(evaluate_args)
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
