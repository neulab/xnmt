import argparse
import sys
import io
import ast

from xnmt.evaluator import BLEUEvaluator, GLEUEvaluator, WEREvaluator, CEREvaluator, RecallEvaluator, ExternalEvaluator
from xnmt.options import OptionParser
from xnmt.xnmt_decode import NO_DECODING_ATTEMPTED

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

def eval_or_empty_list(x):
  try:
    return ast.literal_eval(x)
  except:
    return []

def xnmt_evaluate(ref_file=None, hyp_file=None, evaluator="bleu"):
  """"Returns the eval score (e.g. BLEU) of the hyp sents using reference trg sents
  :param ref_file: path of the reference file
  :param hyp_file: path of the hypothesis trg file
  :param evaluator: Evaluation metrics (bleu/wer/cer)
  """
  args = dict(ref_file=ref_file, hyp_file=hyp_file, evaluator=evaluator)
  cols = args["evaluator"].split("|")
  eval_type  = cols[0]
  eval_param = {} if len(cols) == 1 else {key: value for key, value in [param.split("=") for param in cols[1].split()]}

  hyp_postprocess = lambda line: line.split()
  ref_postprocess = lambda line: line.split()
  if eval_type == "bleu":
    ngram = int(eval_param.get("ngram", 4))
    evaluator = BLEUEvaluator(ngram=int(ngram))
  elif eval_type == "gleu":
    min_len = int(eval_param.get("min", 1))
    max_len = int(eval_param.get("max", 4))
    evaluator = GLEUEvaluator(min_length=min_len, max_length=max_len)
  elif eval_type == "wer":
    evaluator = WEREvaluator()
  elif eval_type == "cer":
    evaluator = CEREvaluator()
  elif eval_type == "recall":
    nbest = int(eval_param.get("nbest", 5))
    hyp_postprocess = lambda x: eval_or_empty_list(x)
    ref_postprocess = lambda x: int(x)
    evaluator = RecallEvaluator(nbest=int(nbest))
  elif eval_type == "mean_avg_precision":
    nbest = int(eval_param.get("nbest", 5))
    hyp_postprocess = lambda x: ast.literal_eval(x)
    ref_postprocess = lambda x: int(x)
    evaluator = MeanAvgPrecisionEvaluator(nbest=int(nbest))
  elif eval_type == 'external':
    path = eval_param.get("path", None)
    higher_better = eval_param.get("higher_better", True)
    if path == None:
      print ("no path given for external evaluation script.")
      return None
    evaluator = ExternalEvaluator(path=path, higher_better=higher_better)

  else:
    raise RuntimeError("Unknown evaluation metric {}".format(eval_type))

  ref_corpus = read_data(args["ref_file"], post_process=ref_postprocess)
  hyp_corpus = read_data(args["hyp_file"], post_process=hyp_postprocess)
  len_before = len(hyp_corpus)
  ref_corpus, hyp_corpus = zip(*filter(lambda x: NO_DECODING_ATTEMPTED not in x[1], zip(ref_corpus, hyp_corpus)))
  if len(ref_corpus) < len_before:
    print("> ignoring %s out of %s test sentences." % (len_before - len(ref_corpus), len_before))

  eval_score = evaluator.evaluate(ref_corpus, hyp_corpus)
  return eval_score

if __name__ == "__main__":

  parser = OptionParser()
  args = parser.args_from_command_line("evaluate", sys.argv[1:])

  score = xnmt_evaluate(args)
  print("{} Score = {}".format(args["evaluator"], score))

