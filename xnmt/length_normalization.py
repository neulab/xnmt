from __future__ import division, generators

import numpy as np
from scipy.stats import norm

from xnmt.serializer import Serializable

class LengthNormalization(object):
  '''
  A template class to generate translation from the output probability model.
  '''
  def normalize_completed(self, completed_hyps, src_length=None):
    """
    normalization step applied to completed hypotheses after search
    :param completed hyps: list of completed Hypothesis objects, will be normalized in-place
    :param src_length: length of source sequence (None if not given)
    :returns: None
    """
    raise NotImplementedError('normalize_completed must be implemented in LengthNormalization subclasses')
  def normalize_partial(self, score_so_far, score_to_add, new_len):
    """
    :param score_so_far:
    :param score_to_add:
    :param new_len: length of output hyp with current word already appended
    :returns: new score after applying score_to_add to score_so_far
    normalization step applied during the search
    """
    return score_so_far + score_to_add # default behavior: add up the log probs

class NoNormalization(LengthNormalization, Serializable):
  '''
  Adding no form of length normalization
  '''
  yaml_tag = u'!NoNormalization'
  def normalize_completed(self, completed_hyps, src_length=None):
    pass


class AdditiveNormalization(LengthNormalization, Serializable):
  '''
  Adding a fixed word penalty everytime the word is added.
  '''
  yaml_tag = u'!AdditiveNormalization'

  def __init__(self, penalty=-0.1, apply_during_search=False):
    self.penalty = penalty
    self.apply_during_search = apply_during_search

  def normalize_completed(self, completed_hyps, src_length=None):
    if not self.apply_during_search:
      for hyp in completed_hyps:
        hyp.score += (len(hyp.id_list) * self.penalty)
  def normalize_partial(self, score_so_far, score_to_add, new_len):
    return score_so_far + score_to_add + (self.penalty if self.apply_during_search else 0.0)


class PolynomialNormalization(LengthNormalization, Serializable):
  '''
  Dividing by the length (raised to some power (default 1))
  '''
  yaml_tag = u'!PolynomialNormalization'

  def __init__(self, m=1, apply_during_search=False):
    self.m = m
    self.apply_during_search = apply_during_search

  def normalize_completed(self, completed_hyps, src_length=None):
    if not self.apply_during_search:
      for hyp in completed_hyps:
        hyp.score /= pow(len(hyp.id_list), self.m)
  def normalize_partial(self, score_so_far, score_to_add, new_len):
    if self.apply_during_search:
      return (score_so_far * pow(new_len-1, self.m) + score_to_add) / pow(new_len, self.m)
    else:
      return score_so_far + score_to_add


class MultinomialNormalization(LengthNormalization, Serializable):
  '''
  The algorithm followed by:
  Tree-to-Sequence Attentional Neural Machine Translation
  https://arxiv.org/pdf/1603.06075.pdf
  '''
  yaml_tag = u'!MultinomialNormalization'

  def __init__(self, sent_stats):
    self.stats = sent_stats

  def trg_length_prob(self, src_length, trg_length):
    v = len(self.stats.src_stat)
    if src_length in self.stats.src_stat:
      src_stat = self.stats.src_stat.get(src_length)
      return (src_stat.trg_len_distribution.get(trg_length, 0) + 1) / (src_stat.num_sents + v)
    return 1

  def normalize_completed(self, completed_hyps, src_length=None):
    """
    :type src_length: length of the src sent
    """
    assert (src_length is not None), "Length of Source Sentence is required"
    for hyp in completed_hyps:
      hyp.score += np.log(self.trg_length_prob(src_length, len(hyp.id_list)))


class GaussianNormalization(LengthNormalization, Serializable):
  '''
   The Gaussian regularization encourages the inference
   to select sents that have similar lengths as the
   sents in the training set.
   refer: https://arxiv.org/pdf/1509.04942.pdf
  '''
  yaml_tag = u'!GaussianNormalization'
  def __init__(self, sent_stats):
    self.stats = sent_stats.trg_stat
    self.num_sent = sent_stats.num_pair
    self.fit_distribution()

  def fit_distribution(self):
    y = np.zeros(self.num_sent)
    iter = 0
    for key in self.stats:
      iter_end = self.stats[key].num_sents + iter
      y[iter:iter_end] = key
      iter = iter_end
    mu, std = norm.fit(y)
    self.distr = norm(mu, std)

  def trg_length_prob(self, trg_length):
    return self.distr.pdf(trg_length)

  def normalize_completed(self, completed_hyps, src_length=None):
    for hyp in completed_hyps:
      hyp.score /= self.trg_length_prob(len(hyp.id_list))
