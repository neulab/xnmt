import argparse
import sys
import io
import ast
from evaluator import BLEUEvaluator, WEREvaluator, CEREvaluator, RecallEvaluator
from options import Option, OptionParser
from xnmt_decode import NO_DECODING_ATTEMPTED

options = [
  Option("ref_file", help_str="path of the reference file"),
  Option("hyp_file", help_str="path of the hypothesis trg file"),
  Option("evaluator", default_value="bleu", help_str="Evaluation metrics (bleu/wer/cer)")
]

def read_data(loc_, post_process=None):
  """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
  """
  data = list()
  with io.open(loc_, encoding='utf-8') as fp:
    for line in fp:
      t = line.strip()
      if post_process is not None:
        t = post_process(t)
      data.append(t)
  return data

def xnmt_evaluate(args):
  """"Returns the eval score (e.g. BLEU) of the hyp sents using reference trg sents
  """
  cols = args.evaluator.split("|")
  eval_type  = cols[0]
  eval_param = {} if len(cols) == 1 else {key: value for key, value in [param.split("=") for param in cols[1].split()]}

  hyp_postprocess = lambda line: line.split()
  ref_postprocess = lambda line: line.split()
  if eval_type == "bleu":
    ngram = int(eval_param.get("ngram", 4))
    evaluator = BLEUEvaluator(ngram=int(ngram))
  elif eval_type == "wer":
    evaluator = WEREvaluator()
  elif eval_type == "cer":
    evaluator = CEREvaluator()
  elif eval_type == "recall":
    nbest = int(eval_param.get("nbest", 5))
    hyp_postprocess = lambda x: ast.literal_eval(x)
    ref_postprocess = lambda x: int(x)
    evaluator = RecallEvaluator(nbest=int(nbest))
  elif eval_type == "mean_avg_precision":
    nbest = int(eval_param.get("nbest", 5))
    hyp_postprocess = lambda x: ast.literal_eval(x)
    ref_postprocess = lambda x: int(x)
    evaluator = MeanAvgPrecisionEvaluator(nbest=int(nbest))
  else:
    raise RuntimeError("Unknown evaluation metric {}".format(eval_type))

  ref_corpus = read_data(args.ref_file, post_process=ref_postprocess)
  hyp_corpus = read_data(args.hyp_file, post_process=hyp_postprocess)
  len_before = len(hyp_corpus)
  ref_corpus, hyp_corpus = zip(*filter(lambda x: NO_DECODING_ATTEMPTED not in x[1], zip(ref_corpus, hyp_corpus)))
  if len(ref_corpus) < len_before:
    print("> ignoring %s out of %s test sentences." % (len_before - len(ref_corpus), len_before))

  eval_score = evaluator.evaluate(ref_corpus, hyp_corpus)
  return eval_score

if __name__ == "__main__":

  parser = OptionParser()
  parser.add_task("evaluate", options)
  args = parser.args_from_command_line("evaluate", sys.argv[1:])

  score = xnmt_evaluate(args)
  print("{} Score = {}".format(args.evaluator, score))

