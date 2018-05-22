"""
This module contains classes for computing evaluation metrics and corresponding classes that contain resulting scores.
"""

from collections import defaultdict, Counter
import math
import subprocess
from typing import List, Sequence, Dict, Tuple, Union, Any, Optional

import numpy as np

from xnmt import logger
from xnmt.persistence import serializable_init, Serializable

class EvalScore(object):
  def higher_is_better(self):
    raise NotImplementedError()
  def value(self):
    raise NotImplementedError()
  def metric_name(self):
    raise NotImplementedError()
  def score_str(self):
    raise NotImplementedError()
  def better_than(self, another_score):
    if another_score is None or another_score.value() is None: return True
    elif self.value() is None: return False
    assert type(self) == type(another_score)
    if self.higher_is_better():
      return self.value() > another_score.value()
    else:
      return self.value() < another_score.value()
  def __str__(self):
    desc = getattr(self, "desc", None)
    if desc:
      return f"{self.metric_name()} ({desc}): {self.score_str()}"
    else:
      return f"{self.metric_name()}: {self.score_str()}"

class LossScore(EvalScore, Serializable):
  yaml_tag = "!LossScore"

  @serializable_init
  def __init__(self, loss, loss_stats=None, desc=None):
    self.loss = loss
    self.loss_stats = loss_stats
    self.desc = desc
    self.serialize_params = {"loss":loss}
    if desc is not None: self.serialize_params["desc"] = desc
    if loss_stats is not None: self.serialize_params["loss_stats"] = desc
  def value(self): return self.loss
  def metric_name(self): return "Loss"
  def higher_is_better(self): return False
  def score_str(self):
    if self.loss_stats is not None and len(self.loss_stats) > 1:
      return "{" + ", ".join("%s: %.5f" % (k, v) for k, v in self.loss_stats.items()) + "}"
    else:
      return f"{self.value():.3f}"

class BLEUScore(EvalScore, Serializable):
  yaml_tag = "!BLEUScore"

  @serializable_init
  def __init__(self, bleu, frac_score_list=None, brevity_penalty_score=None, hyp_len=None, ref_len=None, ngram=4,
               desc=None):
    self.bleu = bleu
    self.frac_score_list = frac_score_list
    self.brevity_penalty_score = brevity_penalty_score
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.ngram   = ngram
    self.desc = desc
    self.serialize_params = {"bleu":bleu, "ngram":ngram}
    self.serialize_params.update(
      {k: getattr(self, k) for k in ["frac_score_list", "brevity_penalty_score", "hyp_len", "ref_len", "desc"] if
       getattr(self, k) is not None})

  def value(self): return self.bleu if self.bleu is not None else 0.0
  def metric_name(self): return "BLEU" + str(self.ngram)
  def higher_is_better(self): return True
  def score_str(self):
    if self.bleu is None:
      return "0"
    else:
      return f"{self.bleu}, {'/'.join(self.frac_score_list)} (BP = {self.brevity_penalty_score:.6f}, " \
             f"ratio={self.hyp_len / self.ref_len:.2f}, hyp_len={self.hyp_len}, ref_len={self.ref_len})"

class GLEUScore(EvalScore, Serializable):
  yaml_tag = "!GLEUScore"

  @serializable_init
  def __init__(self, gleu, hyp_len, ref_len, desc=None):
    self.gleu = gleu
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.desc = desc
    self.serialize_params = {"gleu":gleu, "hyp_len":hyp_len,"ref_len":ref_len}
    if desc is not None: self.serialize_params["desc"] = desc

  def value(self): return self.gleu
  def metric_name(self): return "GLEU"
  def higher_is_better(self): return True
  def score_str(self):
    return "{:.6f}".format(self.value())

class WERScore(EvalScore, Serializable):
  yaml_tag = "!WERScore"

  @serializable_init
  def __init__(self, wer, hyp_len, ref_len, desc=None):
    self.wer = wer
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.desc = desc
    self.serialize_params = {"wer":wer, "hyp_len":hyp_len,"ref_len":ref_len}
    if desc is not None: self.serialize_params["desc"] = desc
  def value(self): return self.wer
  def metric_name(self): return "WER"
  def higher_is_better(self): return False
  def score_str(self):
    return f"{self.value()*100.0:.2f}% ( hyp_len={self.hyp_len}, ref_len={self.ref_len} )"

class CERScore(WERScore, Serializable):
  yaml_tag = "!CERScore"

  @serializable_init
  def __init__(self, cer, hyp_len, ref_len, desc=None):
    self.cer = cer
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.desc = desc
    self.serialize_params = {"cer":cer, "hyp_len":hyp_len,"ref_len":ref_len}
    if desc is not None: self.serialize_params["desc"] = desc
  def metric_name(self): return "CER"
  def value(self): return self.cer

class RecallScore(WERScore, Serializable):
  yaml_tag = "!RecallScore"

  @serializable_init
  def __init__(self, recall, hyp_len, ref_len, nbest=5, desc=None):
    self.recall  = recall
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.nbest   = nbest
    self.desc = desc
    self.serialize_params = {"recall":recall, "hyp_len":hyp_len,"ref_len":ref_len, "nbest":nbest}
    if desc is not None: self.serialize_params["desc"] = desc

  def score_str(self):
    return "{:.2f}%".format(self.value() * 100.0)

  def value(self):
    return self.recall

  def metric_name(self):
    return "Recall" + str(self.nbest)

class ExternalScore(EvalScore, Serializable):
  yaml_tag = "!ExternalScore"

  @serializable_init
  def __init__(self, value, higher_is_better=True, desc=None):
    self.value = value
    self.higher_is_better = higher_is_better
    self.desc = desc
    self.serialize_params = {"value":value, "higher_is_better":higher_is_better}
    if desc is not None: self.serialize_params["desc"] = desc
  def value(self): return self.value
  def metric_name(self): return "External"
  def higher_is_better(self): return self.higher_is_better
  def score_str(self):
    return "{:.3f}".format(self.value)

class SequenceAccuracyScore(EvalScore, Serializable):
  yaml_tag = "!SequenceAccuracyScore"

  @serializable_init
  def __init__(self, accuracy, desc=None):
    self.accuracy = accuracy
    self.desc = desc
    self.serialize_params = {"accuracy":accuracy}
    if desc is not None: self.serialize_params["desc"] = desc
  def higher_is_better(self): return True
  def value(self): return self.accuracy
  def metric_name(self): return "SequenceAccuracy"
  def score_str(self):
    return f"{self.value()*100.0:.2f}%"


class Evaluator(object):
  """
  A class to evaluate the quality of output.
  """

  def evaluate(self, ref: Sequence, hyp: Sequence, desc: Optional[Any] = None) -> EvalScore:
    """
  Calculate the quality of output given a reference.

  Args:
    ref: list of reference sents ( a sentence is a list of tokens )
    hyp: list of hypothesis sents ( a sentence is a list of tokens )
    desc: optional description that is passed on to score objects
  Returns:
  """
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')

  def evaluate_multi(self, ref: Sequence[Sequence], hyp: Sequence, desc: Optional[Any] = None) -> EvalScore:
    """
  Calculate the quality of output given multiple references.

  Args:
    ref: list of tuples of reference sentences ( a sentence is a list of tokens )
    hyp: list of hypothesis sentences ( a sentence is a list of tokens )
    desc: optional description that is passed on to score objects
  """
    raise NotImplementedError(f'evaluate_multi() is not implemented for {type(self)}.')

  def metric_name(self) -> str:
    """
  Return:
    metric name
  """
    raise NotImplementedError('metric_name must be implemented in Evaluator subclasses')

class FastBLEUEvaluator(Evaluator, Serializable):
  """
  Class for computing BLEU scores using a fast Cython implementation.

  Does not support multiple references.
  BLEU scores are computed according to K Papineni et al "BLEU: a method for automatic evaluation of machine translation"

  Args:
    ngram: consider ngrams up to this order (usually 4)
    smooth:
  """
  yaml_tag = "!FastBLEUEvaluator"

  @serializable_init
  def __init__(self, ngram: int = 4, smooth = 0):
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.smooth = smooth
    self.reference_corpus = None
    self.candidate_corpus = None

  def metric_name(self):
    return f"BLEU{self.ngram} score"

  def evaluate(self, ref, hyp, desc=None):
    try:
      from xnmt.cython import xnmt_cython
    except:
      logger.error("BLEU evaluate fast requires xnmt cython installation step."
                   "please check the documentation.")
      raise
    return xnmt_cython.bleu_sentence(self.ngram, self.smooth, ref, hyp)


class BLEUEvaluator(Evaluator, Serializable):
  """
  Compute BLEU scores against one or several references.

  BLEU scores are computed according to K Papineni et al "BLEU: a method for automatic evaluation of machine translation"

  Args:
    ngram: consider ngrams up to this order (usually 4)
  """
  yaml_tag = "!BLEUEvaluator"

  @serializable_init
  def __init__(self, ngram: int = 4):
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.reference_corpus = None
    self.candidate_corpus = None

  def metric_name(self):
    return f"BLEU{self.ngram} score"

  def evaluate(self, ref: Sequence[Sequence[str]], hyp: Sequence[Sequence[str]], desc: Any = None) -> BLEUScore:
    """
    Args:
      ref: reference sentences (single-reference case: sentence is list of strings;
      hyp: list of hypothesis sentences ( a sentence is a list of tokens )
      desc: description to pass on to returned score
    Return:
      Score, including intermediate results such as ngram ratio, sentence length, brevity penalty
    """
    return self._eval(ref, hyp, is_multi_ref=False, desc=desc)

  def evaluate_multi(self, ref: Sequence[Sequence[Sequence[str]]], hyp: Sequence[Sequence[str]],
               desc: Any = None) -> BLEUScore:
    """
    Args:
      ref: list of tuples of reference sentences ( a sentence is a list of tokens )
      hyp: list of hypothesis sentences ( a sentence is a list of tokens )
      desc: optional description that is passed on to score objects
    Return:
      Score, including intermediate results such as ngram ratio, sentence length, brevity penalty
    """
    return self._eval(ref, hyp, is_multi_ref=True, desc=desc)

  def _eval(self, ref: Sequence[Union[Sequence[str], Sequence[Sequence[str]]]], hyp: Sequence[Sequence[str]],
            is_multi_ref: bool, desc: Any = None) -> BLEUScore:
    self.reference_corpus = ref
    self.candidate_corpus = hyp

    assert (len(self.reference_corpus) == len(self.candidate_corpus)), \
           "Length of Reference Corpus and Candidate Corpus should be same"

    # Modified Precision Score
    clipped_ngram_count = Counter()
    candidate_ngram_count = Counter()

    # Brevity Penalty variables
    word_counter = Counter()

    for ref_sent, can_sent in zip(self.reference_corpus, self.candidate_corpus):
      word_counter['candidate'] += len(can_sent)
      if not is_multi_ref:
        word_counter['reference'] += len(ref_sent)

        clip_count_dict, full_count_dict = self._modified_precision(ref_sent, can_sent)

      else:
        ref_lens = sorted([(len(ref_sent_i), abs(len(ref_sent_i) - len(can_sent))) for ref_sent_i in ref_sent],
                          key=lambda x: (x[1],x[0]))
        word_counter['reference'] += ref_lens[0][0]
        counts = [self._modified_precision(ref_sent_i, can_sent) for ref_sent_i in ref_sent]
        full_count_dict = counts[0][1]
        clip_count_dict = defaultdict(Counter)
        for ngram_type in candidate_ngram_count:
          for i in range(len(counts)):
            clip_count_dict[ngram_type] |= counts[i][0][ngram_type]

      for ngram_type in full_count_dict:
        if ngram_type in clip_count_dict:
          clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
        candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

    # Edge case
    # Return 0 if there are no matching n-grams
    # If there are no unigrams, return BLEU score of 0
    # No need to check for higher order n-grams
    if clipped_ngram_count[1] == 0:
      return BLEUScore(bleu=None, ngram=self.ngram, desc=desc)

    frac_score_list = list()
    log_precision_score = 0.
    # Precision Score Calculation
    for ngram_type in range(1, self.ngram + 1):
      frac_score = 0
      if clipped_ngram_count[ngram_type] == 0:
        log_precision_score += -1e10
      else:
        frac_score = clipped_ngram_count[ngram_type] / candidate_ngram_count[ngram_type]
        log_precision_score += self.weights[ngram_type - 1] * math.log(frac_score)
      frac_score_list.append("%.6f" % frac_score)

    precision_score = math.exp(log_precision_score)

    # Brevity Penalty Score
    brevity_penalty_score = self._brevity_penalty(word_counter['reference'], word_counter['candidate'])

    # BLEU Score
    bleu_score = brevity_penalty_score * precision_score
    return BLEUScore(bleu_score, frac_score_list, brevity_penalty_score, word_counter['candidate'], word_counter['reference'], ngram=self.ngram, desc=desc)

  def _brevity_penalty(self, r: int, c: int) -> float:
    """
    Args:
      r: number of words in reference corpus
      c: number of words in candidate corpus
    Return:
      brevity penalty score
    """

    penalty = 1.

    # If candidate sent length is 0 (empty), return 0.
    if c == 0:
      return 0.
    elif c <= r:
      penalty = np.exp(1. - (r / c))
    return penalty

  def _extract_ngrams(self, tokens: Sequence[str]) -> Dict[int, Counter]:
    """
    Extracts ngram counts from the input string

    Args:
      tokens: tokens of string for which the ngram is to be computed
    Return:
      a Counter object containing ngram counts
    """

    ngram_count = defaultdict(Counter)
    num_words = len(tokens)

    for i, first_token in enumerate(tokens[0: num_words]):
      for j in range(0, self.ngram):
        outer_range = i + j + 1
        ngram_type = j + 1
        if outer_range <= num_words:
          ngram_tuple = tuple(tokens[i: outer_range])
          ngram_count[ngram_type][ngram_tuple] += 1

    return ngram_count

  def _modified_precision(self, reference_sent: List[str], candidate_sent: List[str]) \
          -> Tuple[Dict[int,Counter],Dict[int,Counter]]:
    """
    Computes counts useful in modified precision calculations

    Args:
      reference_sent: iterable of tokens
      candidate_sent: iterable of tokens
    Return: tuple of Counter objects
    """

    clipped_ngram_count = defaultdict(Counter)

    reference_ngram_count = self._extract_ngrams(reference_sent)
    candidate_ngram_count = self._extract_ngrams(candidate_sent)

    for ngram_type in candidate_ngram_count:
      clipped_ngram_count[ngram_type] = candidate_ngram_count[ngram_type] & reference_ngram_count[ngram_type]

    return clipped_ngram_count, candidate_ngram_count

class GLEUEvaluator(Evaluator, Serializable):
  """
  Class for computing GLEU Scores.

  Does not support multiple references.
  """
  yaml_tag = "!GLEUEvaluator"
  @serializable_init
  def __init__(self, min_length=1, max_length=4):
    self.min = min_length
    self.max = max_length

  def metric_name(self):
    return f"GLEU{self.ngram}"

  def extract_all_ngrams(self, tokens):
    """
    Extracts ngram counts from the input string

    Args:
      tokens: tokens of string for which the ngram is to be computed
    Return:
      a Counter object containing ngram counts for self.min <= n <= self.max
    """
    num_words = len(tokens)
    ngram_count = Counter()
    for i, first_token in enumerate(tokens[0: num_words]):
      for n in range(self.min, self.max + 1):
        outer_range = i + n
        if outer_range <= num_words:
          ngram_tuple = tuple(tokens[i: outer_range])
          ngram_count[ngram_tuple] += 1
    return ngram_count

  def evaluate(self, ref, hyp, desc=None):
    """
    Args:
      ref: list of reference sents ( a sent is a list of tokens )
      hyp: list of hypothesis sents ( a sent is a list of tokens )
      desc: description to pass on to returned score
    Return:
      Formatted string having GLEU Score
    """
    assert (len(ref) == len(hyp)), \
      "Length of Reference Corpus and Candidate Corpus should be same"
    corpus_n_match = 0
    corpus_total = 0

    total_ref_len, total_hyp_len = 0, 0
    for ref_sent, hyp_sent in zip(ref, hyp):
      total_hyp_len += len(ref_sent)
      total_ref_len += len(hyp_sent)

      hyp_ngrams = self.extract_all_ngrams(hyp_sent)
      tot_ngrams_hyp = sum(hyp_ngrams.values())
      ref_ngrams = self.extract_all_ngrams(ref_sent)
      tot_ngrams_ref = sum(ref_ngrams.values())

      overlap_ngrams = ref_ngrams & hyp_ngrams
      n_match = sum(overlap_ngrams.values())
      n_total = max(tot_ngrams_hyp, tot_ngrams_ref)

      corpus_n_match += n_match
      corpus_total += n_total

    if corpus_total == 0:
      gleu_score = 0.0
    else:
      gleu_score = corpus_n_match / corpus_total
    return GLEUScore(gleu_score, total_ref_len, total_hyp_len, desc=desc)


class WEREvaluator(Evaluator, Serializable):
  """
  A class to evaluate the quality of output in terms of word error rate.

  Does not support multiple references.

  Args:
    case_sensitive: whether scoring should be case-sensitive
    cross_lines: if True, merge all lines into a single line before scoring
                 (careful with long files, quadratic time and space complexity!)
  """
  yaml_tag = "!WEREvaluator"
  @serializable_init
  def __init__(self, case_sensitive: bool = False, cross_lines: bool = False):
    self.case_sensitive = case_sensitive
    self.cross_lines = cross_lines

  def metric_name(self):
    return "Word error rate"

  def evaluate(self, ref, hyp, desc=None):
    """
    Calculate the word error rate of output given a references.

    Args:
      ref: list of list of reference words
      hyp: list of list of decoded words
      desc: description to pass on to returned score
    Return:
      formatted string (word error rate: (ins+del+sub) / (ref_len), plus more statistics)
    """
    if self.cross_lines:
      ref = [sum(ref, [])]
      hyp = [sum(hyp, [])]
    total_dist, total_ref_len, total_hyp_len = 0, 0, 0
    for ref_sent, hyp_sent in zip(ref, hyp):
      dist = self.dist_one_pair(ref_sent, hyp_sent)
      total_dist += dist
      total_ref_len += len(ref_sent)
      total_hyp_len += len(hyp_sent)
    wer_score = float(total_dist) / total_ref_len
    return WERScore(wer_score, total_hyp_len, total_ref_len, desc=desc)

  def dist_one_pair(self, ref_sent, hyp_sent):
    """
    Return:
      tuple (levenshtein distance, reference length)
    """
    if not self.case_sensitive:
      hyp_sent = [w.lower() for w in hyp_sent]
    if not self.case_sensitive:
      ref_sent = [w.lower() for w in ref_sent]
    return -self.seq_sim(ref_sent, hyp_sent)

  # gap penalty:
  gapPenalty = -1.0
  gapSymbol = None

  # similarity function:
  def sim(self, word1, word2):
    """
    Args:
      word1:
      word2:

    Returns:
      float
    """
    if word1 == word2:
      return 0
    else:
      return -1

  def seq_sim(self, l1, l2):
    # compute matrix
    F = [[0] * (len(l2) + 1) for _ in range((len(l1) + 1))]
    for i in range(len(l1) + 1):
      F[i][0] = i * self.gapPenalty
    for j in range(len(l2) + 1):
      F[0][j] = j * self.gapPenalty
    for i in range(0, len(l1)):
      for j in range(0, len(l2)):
        match = F[i][j] + self.sim(l1[i], l2[j])
        delete = F[i][j + 1] + self.gapPenalty
        insert = F[i + 1][j] + self.gapPenalty
        F[i + 1][j + 1] = max(match, delete, insert)
    return F[len(l1)][len(l2)]

class CEREvaluator(Evaluator, Serializable):
  """
  A class to evaluate the quality of output in terms of character error rate.

  Does not support multiple references.

  Args:
    case_sensitive: whether scoring should be case-sensitive
    cross_lines: if True, merge all lines into a single line before scoring
                 (careful with long files, quadratic time and space complexity!)
  """
  yaml_tag = "!CEREvaluator"

  @serializable_init
  def __init__(self, case_sensitive=False, cross_lines=False):
    self.wer_evaluator = WEREvaluator(case_sensitive=case_sensitive, cross_lines=cross_lines)

  def metric_name(self):
    return "Character error rate"

  def evaluate(self, ref, hyp, desc=None):
    """
    Calculate the quality of output given a references.

    Args:
      ref: list of list of reference words
      hyp: list of list of decoded words
      desc: description to pass on to returned score
    Return:
      character error rate: (ins+del+sub) / (ref_len)
    """
    ref_char = [list("".join(ref_sent)) for ref_sent in ref]
    hyp_char = [list("".join(hyp_sent)) for hyp_sent in hyp]
    wer_obj = self.wer_evaluator.evaluate(ref_char, hyp_char)
    return CERScore(wer_obj.value(), wer_obj.hyp_len, wer_obj.ref_len, desc=desc)

class ExternalEvaluator(Evaluator, Serializable):
  """
  A class to evaluate the quality of the output according to an external evaluation script.

  Does not support multiple references.
  The external script should only print a number representing the calculated score.
  """
  yaml_tag = "!ExternalEvaluator"
  @serializable_init
  def __init__(self, path=None, higher_better=True):
    self.path = path
    self.higher_better = higher_better

  def metric_name(self):
    return "External eval script"

  def evaluate(self, ref, hyp, desc=None):
    """
    Calculate the quality of output according to an external script.

    Args:
      ref: (ignored)
      hyp: (ignored)
      desc: description to pass on to returned score
    Return:
      external eval script score
    """
    proc = subprocess.Popen([self.path], stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    external_score = float(out)
    return ExternalScore(external_score, self.higher_better, desc=desc)

class RecallEvaluator(Evaluator,Serializable):
  yaml_tag = "!RecallEvaluator"
  @serializable_init
  def __init__(self, nbest=5):
    self.nbest = nbest

  def metric_name(self):
    return "Recall{}".format(str(self.nbest))

  def evaluate(self, ref, hyp, desc=None):
    true_positive = 0
    for hyp_i, ref_i in zip(hyp, ref):
      if any(ref_i == idx for idx, _ in hyp_i[:self.nbest]):
        true_positive += 1
    score = true_positive / float(len(ref))
    return RecallScore(score, len(hyp), len(ref), nbest=self.nbest, desc=desc)

# The below is needed for evaluating retrieval models, but depends on MeanAvgPrecisionScore which seems to have been
# lost.
#
# class MeanAvgPrecisionEvaluator(object):
#   def __init__(self, nbest=5, desc=None):
#     self.nbest = nbest
#     self.desc = desc
#
#   def metric_name(self):
#     return "MeanAvgPrecision{}".format(str(self.nbest))
#
#   def evaluate(self, ref, hyp):
#     avg = 0
#     for hyp_i, ref_i in zip(hyp, ref):
#         score = 0
#         h = hyp_i[:self.nbest]
#         for x in range(len(h)):
#             if ref_i == h[x][0]:
#                 score = 1/(x+1)
#         avg += score
#     avg = avg/float(len(ref))
#     return MeanAvgPrecisionScore(avg, len(hyp), len(ref), nbest=self.nbest, desc=self.desc)

class SequenceAccuracyEvaluator(Evaluator, Serializable):
  """
  A class to evaluate the quality of output in terms of sequence accuracy.

  Does not support multiple references.
  """
  yaml_tag = "!SequenceAccuracyEvaluator"
  @serializable_init
  def __init__(self, case_sensitive=False):
    self.case_sensitive = case_sensitive

  def metric_name(self):
    return "Sequence accuracy"

  def compare(self, ref_sent, hyp_sent):
    if not self.case_sensitive:
      hyp_sent = [w.lower() for w in hyp_sent]
    if not self.case_sensitive:
      ref_sent = [w.lower() for w in ref_sent]
    return ref_sent == hyp_sent

  def evaluate(self, ref, hyp, desc=None):
    """
    Calculate the accuracy of output given a references.

    Args:
      ref: list of list of reference words
      hyp: list of list of decoded words
      desc: description to pass on to returned score
    Return: formatted string
    """
    correct = sum(self.compare(ref_sent, hyp_sent) for ref_sent, hyp_sent in zip(ref, hyp))
    accuracy = float(correct) / len(ref)
    return SequenceAccuracyScore(accuracy, desc=desc)
