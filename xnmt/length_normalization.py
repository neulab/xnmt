from __future__ import division, generators

import numpy as np
from scipy.stats import norm

class LengthNormalization:
  '''
  A template class to generate translation from the output probability model.
  '''
  def normalize_length(self, completed_hyps, src_length=0):
    raise NotImplementedError('normalize_length must be implemented in LengthNormalization subclasses')


class NoNormalization(LengthNormalization):
  '''
  Adding no form of length normalization
  '''
  def normalize_length(self, completed_hyps, src_length=0):
    pass


class AdditiveNormalization(LengthNormalization):
  '''
  Adding a fixed word penalty everytime the word is added.
  '''
  def __init__(self, penalty=-0.1):
    self.penalty = penalty

  def normalize_length(self, completed_hyps, src_length=0):
    for hyp in completed_hyps:
      hyp.score += (len(hyp.id_list) * self.penalty)


class PolynomialNormalization(LengthNormalization):
  '''
  Dividing by the length (raised to some power (default 1))
  '''
  def __init__(self, m=1):
    self.m = m

  def normalize_length(self, completed_hyps, src_length=0):
    for hyp in completed_hyps:
      hyp.score /= pow(len(hyp.id_list), self.m)


class MultinomialNormalization(LengthNormalization):
  '''
  The algorithm followed by:
  Tree-to-Sequence Attentional Neural Machine Translation
  https://arxiv.org/pdf/1603.06075.pdf
  '''
  def __init__(self, sentence_stats):
    self.stats = sentence_stats

  def trg_length_prob(self, src_length, trg_length):
    v = len(self.stats.src_stat)
    if src_length in self.stats.src_stat:
      src_stat = self.stats.src_stat.get(src_length)
      return (src_stat.trg_len_distribution.get(trg_length, 0) + 1) / (src_stat.num_sentences + v)
    return 1

  def normalize_length(self, completed_hyps, src_length=0):
    """
    :type src_length: length of the src sentence
    """
    assert (src_length > 0), "Length of Source Sentence is required"
    for hyp in completed_hyps:
      hyp.score += np.log(self.trg_length_prob(src_length, len(hyp.id_list)))


class GaussianNormalization(LengthNormalization):
  '''
   The Gaussian regularization encourages the inference
   to select sentences that have similar lengths as the
   sentences in the training set.
   refer: https://arxiv.org/pdf/1509.04942.pdf
  '''
  def __init__(self, sentence_stats):
    self.stats = sentence_stats.trg_stat
    self.num_sent = sentence_stats.num_pair
    self.fit_distribution()

  def fit_distribution(self):
    y = np.zeros(self.num_sent)
    iter = 0
    for key in self.stats:
      iter_end = self.stats[key].num_sentences + iter
      y[iter:iter_end] = key
      iter = iter_end
    mu, std = norm.fit(y)
    self.distr = norm(mu, std)

  def trg_length_prob(self, trg_length):
    return self.distr.pdf(trg_length)

  def normalize_length(self, completed_hyps, src_length=0):
    for hyp in completed_hyps:
      hyp.score /= self.trg_length_prob(len(hyp.id_list))
