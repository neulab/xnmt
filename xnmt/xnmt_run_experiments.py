"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs to files called <experimentname>.log
and <experimentname>.err.log, and reporting on final perplexity metrics.
"""

import argparse
import sys
import os
import xnmt_preproc, xnmt_train, xnmt_decode, xnmt_evaluate
import six
from options import OptionParser, Option
from tee import Tee
import random
import numpy as np


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--dynet-mem", type=int)
  argparser.add_argument("--dynet-seed", type=int)
  argparser.add_argument("--dynet-autobatch", type=int)
  argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
  argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
  argparser.add_argument("--dynet-gpu-ids", type=int)
  argparser.add_argument("--generate-doc", action='store_true', help="Do not run, output documentation instead")
  argparser.add_argument("experiments_file")
  argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
  argparser.set_defaults(generate_doc=False)
  args = argparser.parse_args()

  config_parser = OptionParser()
  config_parser.add_task("preproc", xnmt_preproc.options)
  config_parser.add_task("train", xnmt_train.options)
  config_parser.add_task("decode", xnmt_decode.options)
  config_parser.add_task("evaluate", xnmt_evaluate.options)

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

    preproc_args = exp_tasks["preproc"]

    train_args = exp_tasks["train"]
    train_args.model_file = exp_args.model_file

    decode_args = exp_tasks["decode"]
    decode_args.trg_file = exp_args.hyp_file
    decode_args.model_file = None  # The model is passed to the decoder directly

    evaluate_args = exp_tasks["evaluate"]
    evaluate_args.hyp_file = exp_args.hyp_file
    evaluators = exp_args.eval_metrics.split(",")

    output = Tee(exp_args.out_file, 3)
    err_output = Tee(exp_args.err_file, 3, error=True)

    # Do preprocessing
    print("> Preprocessing")
    xnmt_preproc.xnmt_preproc(preproc_args)

    # Do training
    for task_name in exp_tasks:
      if hasattr(exp_tasks[task_name], "random_search_report"):
        print("> instantiated random parameter search: %s" % exp_tasks[task_name].random_search_report)

    print("> Training")
    xnmt_trainer = xnmt_train.XnmtTrainer(train_args)

    eval_scores = "Not evaluated"
    for i_epoch in six.moves.range(exp_args.run_for_epochs):
      if not exp_args.eval_only:
        xnmt_trainer.run_epoch()

      if xnmt_trainer.early_stopping_reached:
        break

    print('reverting learned weights to best checkpoint..')
    xnmt_trainer.revert_to_best_model()
    if evaluators:
      print("> Evaluating test set")
      output.indent += 2
      xnmt_decode.xnmt_decode(decode_args, model_elements=(
        xnmt_trainer.corpus_parser, xnmt_trainer.model))
      eval_scores = []
      for evaluator in evaluators:
        evaluate_args.evaluator = evaluator
        eval_score = xnmt_evaluate.xnmt_evaluate(evaluate_args)
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
      print("{:<30}| {:<40}".format(experiment_name if i==0 else "", eval_scores[i]))
