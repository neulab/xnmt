import numpy as np

class LengthNormalization:
  '''
  A template class to generate translation from the output probability model.
  '''
  def normalize_length(self, completed_hypotheses, source_length=0):
    raise NotImplementedError('normalize_length must be implemented in LengthNormalization subclasses')


class NoNormalization(LengthNormalization):
  '''
  Adding no form of length normalization
  '''
  def normalize_length(self, completed_hypotheses, source_length=0):
    pass


class AdditiveNormalization(LengthNormalization):
  '''
  Adding a fixed word penalty everytime the word is added.
  '''
  def __init__(self, penalty=-0.1):
    self.penalty = penalty

  def normalize_length(self, completed_hypotheses, source_length=0):
    for hypothesis in completed_hypotheses:
      hypothesis.score += (len(hypothesis.id_list) * self.penalty)


class PolynomialNormalization(LengthNormalization):
  '''
  Dividing by the length (raised to some power (default 1))
  '''
  def __init__(self, m=1):
    self.m = m

  def normalize_length(self, completed_hypotheses, source_length=0):
    for hypothesis in completed_hypotheses:
      hypothesis.score /= pow(len(hypothesis.id_list), self.m)


class MultinomialNormalization(LengthNormalization):
  '''
  The algorithm followed by:
  Tree-to-Sequence Attentional Neural Machine Translation
  https://arxiv.org/pdf/1603.06075.pdf
  '''
  def __init__(self, sentence_stats):
    self.stats = sentence_stats

  def target_length_prob(self, source_length, target_length):
    v = len(self.stats.source_stat)
    if source_length in self.stats.source_stat:
      source_stat = self.stats.source_stat.get(source_length)
      return (source_stat.target_len_distribution.get(target_length, 0) + 1) / (source_stat.num_sentences + v)
    return 1

  def normalize_length(self, completed_hypotheses, source_length=0):
    """
    :type source_length: length of the source sentence
    """
    assert (source_length > 0), "Length of Source Sentence is required"
    for hypothesis in completed_hypotheses:
      hypothesis.score += np.log(self.target_length_prob(source_length, len(hypothesis.id_list)))

