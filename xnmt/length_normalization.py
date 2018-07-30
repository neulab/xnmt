from numbers import Real
from typing import Sequence, Optional

import numpy as np
from scipy.stats import norm

from xnmt.persistence import serializable_init, Serializable
from xnmt import search_strategy
from xnmt.vocab import Vocab

class LengthNormalization(object):
  """
  A template class to adjust scores for length normalization during search.
  """

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    """
    Apply normalization step to completed hypotheses after search and return the normalized scores.
    
    Args:
      completed_hyps: list of completed Hypothesis objects, will be normalized in-place
      src_length: length of source sequence (None if not given)
    Returns:
      normalized scores
    """
    raise NotImplementedError('normalize_completed must be implemented in LengthNormalization subclasses')

  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    """
    Apply normalization step after expanding a partial hypothesis and selecting the top k scores.

    Args:
      score_so_far: log score of the partial hypothesis
      score_to_add: log score of the top-k item that is to be added
      new_len: new length of partial hypothesis with current word already appended
    Returns:
      new score after applying score_to_add to score_so_far
    """
    return score_so_far + score_to_add # default behavior: add up the log probs


class NoNormalization(LengthNormalization, Serializable):
  """
  Adding no form of length normalization.
  """
  yaml_tag = '!NoNormalization'

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    return [hyp.score for hyp in completed_hyps]

class AdditiveNormalization(LengthNormalization, Serializable):
  """
  Adding a fixed word penalty everytime the word is added.
  """
  yaml_tag = '!AdditiveNormalization'

  @serializable_init
  def __init__(self, penalty: Real = -0.1, apply_during_search: bool = False):
    self.penalty = penalty
    self.apply_during_search = apply_during_search

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    if self.apply_during_search:
      return [hyp.score for hyp in completed_hyps]
    else:
      return [hyp.score + (len(hyp.id_list) * self.penalty) for hyp in completed_hyps]
  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    return score_so_far + score_to_add + (self.penalty if self.apply_during_search else 0.0)


class PolynomialNormalization(LengthNormalization, Serializable):
  """
  Dividing by the length (raised to some power)
  """
  yaml_tag = '!PolynomialNormalization'

  @serializable_init
  def __init__(self, m: Real = 1, apply_during_search: bool = False):
    self.m = m
    self.apply_during_search = apply_during_search
    self.pows = []

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    if self.apply_during_search:
      return [hyp.score for hyp in completed_hyps]
    else:
      return [(hyp.score / pow(len(hyp.output.word_ids), self.m)) for hyp in completed_hyps]
  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    if self.apply_during_search:
      self.update_pows(new_len)
      return (score_so_far * self.pows[new_len-1] + score_to_add) / self.pows[new_len]
    else:
      return score_so_far + score_to_add
  def update_pows(self, new_len):
    if len(self.pows) < new_len+1:
      for i in range(len(self.pows), new_len+1):
        self.pows.append(pow(i, self.m))


class MultinomialNormalization(LengthNormalization, Serializable):
  """
  The algorithm followed by:
  Tree-to-Sequence Attentional Neural Machine Translation
  https://arxiv.org/pdf/1603.06075.pdf
  """
  yaml_tag = '!MultinomialNormalization'

  @serializable_init
  def __init__(self, sent_stats):
    self.stats = sent_stats

  def trg_length_prob(self, src_length, trg_length):
    v = len(self.stats.src_stat)
    if src_length in self.stats.src_stat:
      src_stat = self.stats.src_stat.get(src_length)
      return (src_stat.trg_len_distribution.get(trg_length, 0) + 1) / (src_stat.num_sents + v)
    return 1

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    """
    Args:
      completed_hyps:
      src_length: length of the src sent
    """
    assert (src_length is not None), "Length of Source Sentence is required"

    return [hyp.score + np.log(self.trg_length_prob(src_length, len(hyp.id_list))) for hyp in completed_hyps]


class GaussianNormalization(LengthNormalization, Serializable):
  """
   The Gaussian regularization encourages the inference
   to select sents that have similar lengths as the
   sents in the training set.
   refer: https://arxiv.org/pdf/1509.04942.pdf
  """
  yaml_tag = '!GaussianNormalization'

  @serializable_init
  def __init__(self, sent_stats, length_ratio=False, src_cond=False):
    """
     Args:
       sent_stats: A SentenceStats object
       length_ratio: Instead of fitting len(trg) in the training set, fit the length ratio len(trg) / len(src)
       src_cond: Instead of fitting len(trg) in the training set, fit the length of trg with the given len(src) 
    """
    self.sent_stats = sent_stats
    self.num_sent = sent_stats.num_pair
    self.length_ratio = length_ratio
    self.src_cond = src_cond
    self.fit_distribution()

  def fit_distribution(self):
    if self.length_ratio:
      stats = self.sent_stats.src_stat
      num_sent = self.sent_stats.num_pair
      y = np.zeros(num_sent)
      iter = 0
      for key in stats:
        for t_len, count in stats[key].trg_len_distribution.items():
          iter_end = count + iter
          y[iter:iter_end] = t_len / float(key)
          iter = iter_end
      mu, std = norm.fit(y)
      self.distr = norm(mu, std)
    elif self.src_cond:
      stats = self.sent_stats.src_stat
      self.distr = {}
      self.max_key = -1
      for key in stats:
        if key > self.max_key: self.max_key = key
        num_trg = stats[key].num_sents
        y = np.zeros(num_trg)
        iter = 0
        for t_len, count in stats[key].trg_len_distribution.items():
          iter_end = count + iter
          y[iter:iter_end] = t_len
          iter = iter_end
        mu, std = norm.fit(y)
        if std == 0: std = np.sqrt(key)
        self.distr[key] = norm(mu, std)
      for i in range(self.max_key-1, -1, -1):
        if i not in self.distr:
          self.distr[i] = self.distr[i+1]
    else:
      stats = self.sent_stats.trg_stat
      y = np.zeros(self.num_sent)
      curr_iter = 0
      for key in stats:
        iter_end = stats[key].num_sents + curr_iter
        y[curr_iter:iter_end] = key
        curr_iter = iter_end
      mu, std = norm.fit(y)
      self.distr = norm(mu, std)

  def trg_length_prob(self, src_length, trg_length):
    if self.length_ratio:
      assert (src_length is not None), "Length of Source Sentence is required in GaussianNormalization when length_ratio=True"
      return self.distr.pdf(trg_length/src_length)
    elif self.src_cond:
      if src_length in self.distr:
        return self.distr[src_length].pdf(trg_length)
      else:
        return self.distr[self.max_key].pdf(trg_length)
    else:
      return self.distr.pdf(trg_length)

  def normalize_completed(self, completed_hyps: Sequence['search_strategy.BeamSearch.Hypothesis'],
                          src_length: Optional[int] = None) -> Sequence[float]:
    return [hyp.score / self.trg_length_prob(src_len, hyp.length) for hyp, src_len in zip(completed_hyps, src_length)]

class EosBooster(Serializable):
  """
  Callable that applies boosting of end-of-sequence token, can be used with :class:`xnmt.search_strategy.BeamSearch`.

  Args:
    boost_val: value to add to the eos token's log probability. Positive values make sentences shorter, negative values
               make sentences longer.
  """
  yaml_tag = "!EosBooster"
  @serializable_init
  def __init__(self, boost_val: float):
    self.boost_val = boost_val
  def __call__(self, scores:np.ndarray) -> None:
    scores[Vocab.ES] += self.boost_val
