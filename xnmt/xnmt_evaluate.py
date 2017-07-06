import argparse
import sys
import io
from evaluator import BLEUEvaluator, WEREvaluator, CEREvaluator
from options import Option, OptionParser
from xnmt_decode import NO_DECODING_ATTEMPTED

options = [
  Option("ref_file", help_str="path of the reference file"),
  Option("hyp_file", help_str="path of the hypothesis trg file"),
  Option("evaluator", default_value="bleu", help_str="Evaluation metrics (bleu/wer/cer)")
]

def read_data(loc_):
  """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
  """
  data = list()
  with io.open(loc_, encoding='utf-8') as fp:
    for line in fp:
      t = line.split()
      data.append(t)
  return data

def xnmt_evaluate(args):
  """"Returns the eval score (e.g. BLEU) of the hyp sents using reference trg sents
  """

  if args.evaluator == "bleu":
    evaluator = BLEUEvaluator(ngram=4)
  elif args.evaluator == "wer":
    evaluator = WEREvaluator()
  elif args.evaluator == "cer":
    evaluator = CEREvaluator()
  else:
    raise RuntimeError("Unknown evaluation metric {}".format(args.evaluator))

  ref_corpus = read_data(args.ref_file)
  hyp_corpus = read_data(args.hyp_file)
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

