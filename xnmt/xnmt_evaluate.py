import argparse
import sys
from typing import Any, Sequence, Union

from xnmt.evaluator import * # import everything so we can parse it with eval()
from xnmt.inference import NO_DECODING_ATTEMPTED

def read_data(loc_, post_process=None):
  """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
  """
  data = list()
  with open(loc_, encoding='utf-8') as fp:
    for line in fp:
      t = line.strip()
      if post_process is not None:
        t = post_process(t)
      data.append(t)
  return data

eval_shortcuts = {
  "bleu": lambda:BLEUEvaluator(),
  "gleu": lambda:GLEUEvaluator(),
  "wer": lambda:WEREvaluator(),
  "cer": lambda:CEREvaluator(),
  "recall": lambda:RecallEvaluator(),
  "accuracy": lambda:SequenceAccuracyEvaluator(),
}


def xnmt_evaluate(ref_file: Union[str, Sequence[str]], hyp_file: Union[str, Sequence[str]],
                  evaluators: Sequence[Evaluator], desc: Any = None) -> Sequence[EvalScore]:
  """"Returns the eval score (e.g. BLEU) of the hyp sents using reference trg sents

  Args:
    ref_file: path of the reference file
    hyp_file: path of the hypothesis trg file
    evaluators: Evaluation metrics. Can be a list of evaluator objects, or a shortcut string
    desc: descriptive string passed on to evaluators
  """
  hyp_postprocess = lambda line: line.split()
  ref_postprocess = lambda line: line.split()

  ref_corpus = read_data(ref_file, post_process=ref_postprocess)
  hyp_corpus = read_data(hyp_file, post_process=hyp_postprocess)
  len_before = len(hyp_corpus)
  ref_corpus, hyp_corpus = zip(*filter(lambda x: NO_DECODING_ATTEMPTED not in x[1], zip(ref_corpus, hyp_corpus)))
  if len(ref_corpus) < len_before:
    logger.info(f"> ignoring {len_before - len(ref_corpus)} out of {len_before} test sentences.")

  return [evaluator.evaluate(ref_corpus, hyp_corpus, desc=desc) for evaluator in evaluators]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--metric", help=f"Scoring metric(s), a comma-separated string. "
                                      f"Accepted metrics are {', '.join(eval_shortcuts.keys())}. Alternatively, "
                                      f"metrics with non-default settings can by used by specifying a Python list of "
                                      f"Evaluator objects to be parsed using eval(). "
                                      f"Example: '[WEREvaluator(case_sensitive=True)]'")
  parser.add_argument("--hyp", help="Path to read hypothesis file from")
  parser.add_argument("--ref", help="Path to read reference file from")
  args = parser.parse_args()

  evaluators = args.metrics
  try:
    evaluators = [eval_shortcuts[shortcut]() for shortcut in evaluators.split(",")]
  except KeyError:
    evaluators = eval(evaluators)

  scores = xnmt_evaluate(args.ref, args.hyp, evaluators)
  for score in scores:
    print(score)

if __name__ == "__main__":
  sys.exit(main())