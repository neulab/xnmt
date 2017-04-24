from __future__ import division, generators

import numpy as np
from collections import defaultdict, Counter
import math
import warnings


class Evaluator(object):
  """
  A class to evaluate the quality of output.
  """

  def evaluate(self, ref, hyp):
    """
    Calculate the quality of output given a references.
    :param ref:
    :param hyp:
    :return:
    """
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')

  def metric_name(self):
    """
    :return: a string
    """
    raise NotImplementedError('metric_name must be implemented in Evaluator subclasses')


class BLEUEvaluator(Evaluator):
    # Doc to be added
    def __init__(self, ngram=4):
        """
        :param ref_corpus:
        :param can_corpus:
        :param ngram:
        """
        self.ngram = ngram
        self.weights = (1/ngram) * np.ones(ngram, dtype=np.float32)
        self.reference_corpus = None
        self.candidate_corpus = None
    
    def metric_name(self):
        return "BLEU score"

    # Doc to be added
    def evaluate(self, ref, hyp):
        """
        :rtype: object
        :param ref:
        :param hyp:
        :return:
        :return:
        """
        self.reference_corpus = ref
        self.candidate_corpus = hyp

        assert(len(self.reference_corpus) == len(self.candidate_corpus)), \
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
                    clipped_ngram_count[ngram_type] += 0. # This line may not be required

                candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

        # Edge case
        # Return 0 if there are no matching n-grams
        # If there are no unigrams, return BLEU score of 0
        # No need to check for higher order n-grams
        if clipped_ngram_count[1] == 0:
            return 0.

        frac_score_list = list()
        log_precision_score = 0.
        # Precision Score Calculation
        for ngram_type in range(1, self.ngram+1):
            frac_score = 0
            if clipped_ngram_count[ngram_type] == 0:
                log_precision_score += -1e10
            else:
                frac_score = clipped_ngram_count[ngram_type] / candidate_ngram_count[ngram_type]
                log_precision_score += self.weights[ngram_type-1] * math.log(frac_score)
            frac_score_list.append(str(frac_score))

        precision_score = math.exp(log_precision_score)

        # Brevity Penalty Score
        brevity_penalty_score = self.brevity_penalty(word_counter['reference'], word_counter['candidate'])

        # BLEU Score
        bleu_score = brevity_penalty_score * precision_score

        return "BLEU = {}, {}(BP = {}, ratio={}, hyp_len={}, ref_len={})".format(bleu_score,
                                                                                 '/'.join(frac_score_list),
                                                                                 brevity_penalty_score,
                                                                                 word_counter['candidate'] / word_counter['reference'],
                                                                                 word_counter['candidate'],
                                                                                 word_counter['reference'])

    # Doc to be added
    def brevity_penalty(self, r, c):
        """
        :param r:
        :param c:
        :return:
        """

        penalty = 1.

        # If candidate sentence length is 0 (empty), return 0.
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

    # Doc to be added
    def modified_precision(self, reference_sentence, candidate_sentence):
        """
        :param reference_sentence:
        :param candidate_sentence:
        :return:
        """

        clipped_ngram_count = defaultdict(Counter)

        reference_ngram_count = self.extract_ngrams(reference_sentence)
        candidate_ngram_count = self.extract_ngrams(candidate_sentence)

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
    :return: word error rate: (ins+del+sub) / (ref_len)
    """
    total_dist, total_ref_len = 0, 0
    for ref_sent, hyp_sent in zip(ref, hyp):
      dist, ref_len = self.dist_one_pair(ref_sent, hyp_sent)
      total_dist += dist
      total_ref_len += ref_len
    return float(total_dist) / total_ref_len
  def dist_one_pair(self, ref_sent, hyp_sent):
    """
    :return: tuple (levenshtein distance, reference length) 
    """
    if not self.case_sensitive:
      hyp_sent = map(lambda w:w.lower(), hyp_sent)
    if not self.case_sensitive:
      ref_sent = map(lambda w:w.lower(), ref_sent)
    return -self.seq_sim(ref_sent, hyp_sent), len(ref_sent)

  # gap penalty:
  gapPenalty = -1.0
  gapSymbol = None

  # similarity function:
  def sim(self, word1, word2):
    if word1 == word2: return 0
    else: return -1

  def seq_sim(self, l1, l2):
    # compute matrix
    F = [[0] * (len(l2) + 1) for i in xrange((len(l1) + 1))]
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
    return self.wer_evaluator.evaluate(ref_char, hyp_char)


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
    #                                              [candidate1, candidate3]))
