"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs to files called <experimentname>.log
and <experimentname>.err.log, and reporting on final perplexity metrics.
"""

import argparse
import sys
import xnmt_train, xnmt_decode, xnmt_evaluate
from options import OptionParser, Option


class Tee:
  """
  Emulates a standard output or error streams. Calls to write on that stream will result
  in printing to stdout as well as logging to a file.
  """

  def __init__(self, name, indent=0, error=False):
    self.file = open(name, 'w')
    self.stdstream = sys.stderr if error else sys.stdout
    self.indent = indent
    self.error = error
    if error:
      sys.stderr = self
    else:
      sys.stdout = self

  def close(self):
    if self.error:
      sys.stderr = self.stdstream
    else:
      sys.stdout = self.stdstream
    self.file.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def write(self, data):
    self.file.write(data)
    self.stdstream.write(" " * self.indent + data)
    self.flush()

  def flush(self):
    self.file.flush()
    self.stdstream.flush()


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('experiments_file')
  argparser.add_argument('--dynet-mem', type=int)
  argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
  argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
  argparser.add_argument("--generate-doc", action='store_true', help="Do not run, output documentation instead")
  argparser.set_defaults(generate_doc=False)
  args = argparser.parse_args()

  config_parser = OptionParser()
  config_parser.add_task("train", xnmt_train.options)
  config_parser.add_task("decode", xnmt_decode.options)
  config_parser.add_task("evaluate", xnmt_evaluate.options)

  # Tweak the options to make config files less repetitive:
  # - Delete evaluate:evaluator, replace with exp:eval_metrics
  # - Delete decode:hyp_file, evaluate:hyp_file, replace with exp:hyp_file
  # - Delete train:model, decode:model_file, replace with exp:model_file
  config_parser.remove_option("evaluate", "evaluator")
  config_parser.remove_option("decode", "target_file")
  config_parser.remove_option("evaluate", "hyp_file")
  config_parser.remove_option("train", "model_file")
  config_parser.remove_option("decode", "model_file")

  experiment_options = [
    Option("model_file", required=True, help="Location to write the model file"),
    Option("hyp_file", required=True, help="Temporary location to write decoded output for evaluation"),
    Option("eval_metrics", default_value="bleu", help="Comma-separated list of evaluation metrics"),
    Option("run_for_epochs", int, help="How many epochs to run each test for"),
    Option("decode_every", int, default_value=0, help="Evaluation period in epochs. If set to 0, will never evaluate."),
  ]

  config_parser.add_task("experiment", experiment_options)

  if args.generate_doc:
    print config_parser.generate_options_table()
    exit(0)

  config = config_parser.args_from_config_file(args.experiments_file)

  results = []

  for experiment_name, exp_tasks in config.items():
    print("=> Running {}".format(experiment_name))

    exp_args = exp_tasks["experiment"]

    train_args = exp_tasks["train"]
    train_args.model_file = exp_args.model_file

    decode_args = exp_tasks["decode"]
    decode_args.target_file = exp_args.hyp_file
    decode_args.model_file = None  # The model is passed to the decoder directly

    evaluate_args = exp_tasks["evaluate"]
    evaluate_args.hyp_file = exp_args.hyp_file
    evaluators = exp_args.eval_metrics.split(",")

    output = Tee(experiment_name + ".log", 3)
    err_output = Tee(experiment_name + ".err.log", 3, error=True)
    print("> Training")

    xnmt_trainer = xnmt_train.XnmtTrainer(train_args)

    eval_scores = "Not evaluated"
    for i_epoch in xrange(exp_args.run_for_epochs):
      xnmt_trainer.run_epoch()

      if exp_args.decode_every != 0 and (i_epoch+1) % exp_args.decode_every == 0:
        print("> Evaluating")
        xnmt_decode.xnmt_decode(decode_args, model_elements=(
          xnmt_trainer.input_reader.vocab, xnmt_trainer.output_reader.vocab, xnmt_trainer.translator))
        eval_scores = []
        for evaluator in evaluators:
          evaluate_args.evaluator = evaluator
          eval_score = xnmt_evaluate.xnmt_evaluate(evaluate_args)
          print("{}: {}".format(evaluator, eval_score))
          eval_scores.append(eval_score)

        # The temporary file is cleared by xnmt_decode, not clearing it explicitly here allows it to stay around
        # after the experiment is complete.

    results.append((experiment_name, eval_scores))

    output.close()
    err_output.close()

  print("")
  print("{:<20}|{:<40}".format("Experiment", "Final Scores"))
  print("-" * (60 + 1))

  for line in results:
    experiment_name, eval_scores = line
    print("{:<20}|{:<40}".format(experiment_name, eval_scores))
