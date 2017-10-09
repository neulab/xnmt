from __future__ import division, generators
import numpy as np
import math
import six
import functools
from collections import defaultdict, Counter, deque

from xnmt.vocab import Vocab

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
    return "{}: {}".format(self.metric_name(), self.score_str())

class LossScore(EvalScore):
  def __init__(self, loss):
    self.loss = loss
  def value(self): return self.loss
  def metric_name(self): return "Loss"
  def higher_is_better(self): return False
  def score_str(self):
    return "{:.3f}".format(self.value())

class BLEUScore(EvalScore):
  def __init__(self, bleu, frac_score_list=None, brevity_penalty_score=None, hyp_len=None, ref_len=None, ngram=4):
    self.bleu = bleu
    self.frac_score_list = frac_score_list
    self.brevity_penalty_score = brevity_penalty_score
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.ngram   = ngram

  def value(self): return self.bleu
  def metric_name(self): return "BLEU" + str(self.ngram)
  def higher_is_better(self): return True
  def score_str(self):
    if self.bleu is None:
      return "0"
    else:
      return "{}, {} (BP = {:.6f}, ratio={:.2f}, hyp_len={}, ref_len={})".format(self.bleu,
                                                                            '/'.join(self.frac_score_list),
                                                                            self.brevity_penalty_score,
                                                                            self.hyp_len / self.ref_len,
                                                                            self.hyp_len,
                                                                            self.ref_len)

class WERScore(EvalScore):
  def __init__(self, wer, hyp_len, ref_len):
    self.wer = wer
    self.hyp_len = hyp_len
    self.ref_len = ref_len
  def value(self): return self.wer
  def metric_name(self): return "WER"
  def higher_is_better(self): return False
  def score_str(self):
    return "{:.2f}% ( hyp_len={}, ref_len={} )".format(self.value()*100.0, self.hyp_len, self.ref_len)

class CERScore(WERScore):
  def __init__(self, cer, hyp_len, ref_len):
    self.cer = cer
    self.hyp_len = hyp_len
    self.ref_len = ref_len
  def metric_name(self): return "CER"
  def value(self): return self.cer

class RecallScore(WERScore):
  def __init__(self, recall, hyp_len, ref_len, nbest=5):
    self.recall  = recall
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.nbest   = nbest

  def score_str(self):
    return "{:.2f}%".format(self.value() * 100.0)

  def value(self):
    return self.recall

  def metric_name(self):
    return "Recall" + str(self.nbest)

class Evaluator(object):
  """
  A class to evaluate the quality of output.
  """

  def evaluate(self, ref, hyp):
    """
  Calculate the quality of output given a references.
  :param ref: list of reference sents ( a sent is a list of tokens )
  :param hyp: list of hypothesis sents ( a sent is a list of tokens )
  :return:
  """
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')

  def metric_name(self):
    """
  :return: a string
  """
    raise NotImplementedError('metric_name must be implemented in Evaluator subclasses')

  def evaluate_fast(self, ref, hyp):
    raise NotImplementedError('evaluate_fast is not implemented for:', self.__class__.__name__)

class BLEUEvaluator(Evaluator):
  # Class for computing BLEU Scores accroding to
  # K Papineni et al "BLEU: a method for automatic evaluation of machine translation"
  def __init__(self, ngram=4, smooth=0):
    """
    :param ngram: default value of 4 is generally used
    """
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.smooth = smooth
    self.reference_corpus = None
    self.candidate_corpus = None

  def metric_name(self):
    return "BLEU%d score" % (self.ngram)

  def evaluate_fast(self, ref, hyp):
    try:
      from xnmt.cython import xnmt_cython
    except:
      print("BLEU evaluate fast requires xnmt cython installation step.",
            "please check the documentation.")
    return xnmt_cython.bleu_sentence(self.ngram, self.smooth, ref, hyp)

  # Doc to be added
  def evaluate(self, ref, hyp):
    """
    :rtype: object
    :param ref: list of reference sents ( a sent is a list of tokens )
    :param hyp: list of hypothesis sents ( a sent is a list of tokens )
    :return: Formatted string having BLEU Score with different intermediate results such as ngram ratio,
    sent length, brevity penalty
    """
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
      word_counter['reference'] += len(ref_sent)
      word_counter['candidate'] += len(can_sent)

      clip_count_dict, full_count_dict = self.modified_precision(ref_sent, can_sent)

      for ngram_type in full_count_dict:
        if ngram_type in clip_count_dict:
          clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
        else:
          clipped_ngram_count[ngram_type] += 0.  # This line may not be required

        candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

    # Edge case
    # Return 0 if there are no matching n-grams
    # If there are no unigrams, return BLEU score of 0
    # No need to check for higher order n-grams
    if clipped_ngram_count[1] == 0:
      return BLEUScore(bleu=None, ngram=self.ngram)

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
    brevity_penalty_score = self.brevity_penalty(word_counter['reference'], word_counter['candidate'])

    # BLEU Score
    bleu_score = brevity_penalty_score * precision_score
    return BLEUScore(bleu_score, frac_score_list, brevity_penalty_score, word_counter['candidate'], word_counter['reference'], ngram=self.ngram)

  # Doc to be added
  def brevity_penalty(self, r, c):
    """
    :param r: number of words in reference corpus
    :param c: number of words in candidate corpus
    :return: brevity penalty score
    """

    penalty = 1.

    # If candidate sent length is 0 (empty), return 0.
    if c == 0:
      return 0.
    elif c <= r:
      penalty = np.exp(1. - (r / c))
    return penalty

  # Doc to be added
  def extract_ngrams(self, tokens):
    """
    Extracts ngram counts from the input string
    :param tokens: tokens of string for which the ngram is to be computed
    :return: a Counter object containing ngram counts
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

  def modified_precision(self, reference_sent, candidate_sent):
    """
    Computes counts useful in modified precision calculations
    :param reference_sent: iterable of tokens
    :param candidate_sent: iterable of tokens
    :return: tuple of Counter objects
    """

    clipped_ngram_count = defaultdict(Counter)

    reference_ngram_count = self.extract_ngrams(reference_sent)
    candidate_ngram_count = self.extract_ngrams(candidate_sent)

    for ngram_type in candidate_ngram_count:
      clipped_ngram_count[ngram_type] = candidate_ngram_count[ngram_type] & reference_ngram_count[ngram_type]

    return clipped_ngram_count, candidate_ngram_count

class WEREvaluator(Evaluator):
  """
  A class to evaluate the quality of output in terms of word error rate.
  """

  def __init__(self, case_sensitive=False):
    self.case_sensitive = case_sensitive

  def metric_name(self):
    return "Word error rate"

  def evaluate(self, ref, hyp):
    """
    Calculate the word error rate of output given a references.
    :param ref: list of list of reference words
    :param hyp: list of list of decoded words
    :return: formatted string (word error rate: (ins+del+sub) / (ref_len), plus more statistics)
    """
    total_dist, total_ref_len, total_hyp_len = 0, 0, 0
    for ref_sent, hyp_sent in zip(ref, hyp):
      dist = self.dist_one_pair(ref_sent, hyp_sent)
      total_dist += dist
      total_ref_len += len(ref_sent)
      total_hyp_len += len(hyp_sent)
    wer_score = float(total_dist) / total_ref_len
    return WERScore(wer_score, total_hyp_len, total_ref_len)

  def dist_one_pair(self, ref_sent, hyp_sent):
    """
    :return: tuple (levenshtein distance, reference length)
    """
    if not self.case_sensitive:
      hyp_sent = list(six.moves.map(lambda w: w.lower(), hyp_sent))
    if not self.case_sensitive:
      ref_sent = list(six.moves.map(lambda w: w.lower(), ref_sent))
    return -self.seq_sim(ref_sent, hyp_sent)

  # gap penalty:
  gapPenalty = -1.0
  gapSymbol = None

  # similarity function:
  def sim(self, word1, word2):
    if word1 == word2:
      return 0
    else:
      return -1

  def seq_sim(self, l1, l2):
    # compute matrix
    F = [[0] * (len(l2) + 1) for i in range((len(l1) + 1))]
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

class CEREvaluator(object):
  """
  A class to evaluate the quality of output in terms of character error rate.
  """

  def __init__(self, case_sensitive=False):
    self.wer_evaluator = WEREvaluator(case_sensitive=case_sensitive)

  def metric_name(self):
    return "Character error rate"

  def evaluate(self, ref, hyp):
    """
    Calculate the quality of output given a references.
    :param ref: list of list of reference words
    :param hyp: list of list of decoded words
    :return: character error rate: (ins+del+sub) / (ref_len)
    """
    ref_char = [list("".join(ref_sent)) for ref_sent in ref]
    hyp_char = [list("".join(hyp_sent)) for hyp_sent in hyp]
    wer_obj = self.wer_evaluator.evaluate(ref_char, hyp_char)
    return CERScore(wer_obj.value(), wer_obj.hyp_len, wer_obj.ref_len)

if __name__ == "__main__":
  # Example 1
  reference1 = "It is a guide to action that ensures that the military will forever heed Party commands".split()
  candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()

  obj = BLEUEvaluator(ngram=4)
  print("xnmt bleu score :")
  print(obj.evaluate([reference1], [candidate1]))
  # print("nltk BLEU scores"), print(corpus_bleu([[reference1]], [candidate1]))

  # Example 2
  reference2 = "the cat is on the mat".split()
  candidate2 = "the the the the the the the".split()

  # Generates a warning because of no 2-grams and beyond
  obj = BLEUEvaluator(ngram=4)
  print("xnmt bleu score :")
  print(obj.evaluate([reference2], [candidate2]))
  # print("nltk BLEU scores"), print(corpus_bleu([[reference2]], [candidate2]))

  # Example 3 (candidate1 + candidate3)
  reference3 = "he was interested in world history because he read the book".split()
  candidate3 = "he read the book because he was interested in world history".split()
  obj = BLEUEvaluator(ngram=4)
  print("xnmt bleu score :")
  print(obj.evaluate([reference1, reference3], [candidate1, candidate3]))
  # print("nltk BLEU scores"), print(corpus_bleu([[reference1], [reference3]],
  #                        [candidate1, candidate3]))

class RecallEvaluator(object):
  def __init__(self, nbest=5):
    self.nbest = nbest

  def metric_name(self):
    return "Recall{}".format(str(self.nbest))

  def evaluate(self, ref, hyp):
    true_positive = 0
    for hyp_i, ref_i in zip(hyp, ref):
      if any(ref_i == idx for idx, score in hyp_i[:self.nbest]):
        true_positive += 1
    score = true_positive / float(len(ref))
    return RecallScore(score, len(hyp), len(ref), nbest=self.nbest)

class MeanAvgPrecisionEvaluator(object):
  def __init__(self, nbest=5):
    self.nbest = nbest

  def metric_name(self):
    return "MeanAvgPrecision{}".format(str(self.nbest))

  def evaluate(self, ref, hyp):
    avg = 0
    for hyp_i, ref_i in zip(hyp, ref):
        score = 0
        h = hyp_i[:self.nbest]
        for x in range(len(h)):
            if ref_i == h[x][0]:
                score = 1/(x+1)
        avg += score
    avg = avg/float(len(ref))
    return MeanAvgPrecisionScore(avg, len(hyp), len(ref), nbest=self.nbest)
