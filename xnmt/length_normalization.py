
class LengthNormalization:
  '''
  A template class to generate translation from the output probability model.
  '''
  def normalize_length(self, completed_hypotheses):
    raise NotImplementedError('normalize_length must be implemented in LengthNormalization subclasses')


class NoNormalization(LengthNormalization):
  '''
  Adding no form of length normalization
  '''
  def normalize_length(self, completed_hypotheses):
    pass


class AdditiveNormalization(LengthNormalization):
  '''
  Adding a fixed word penalty everytime the word is added.
  '''
  def __init__(self, penalty=-0.1):
    self.penalty = penalty

  def normalize_length(self, completed_hypotheses):
    for hypothesis in completed_hypotheses:
      hypothesis.score += (len(hypothesis.id_list) * self.penalty)


class PolynomialNormalization(LengthNormalization):
  '''
  Dividing by the length (raised to some power (default 1))
  '''
  def __init__(self, m=1):
    self.m = m

  def normalize_length(self, completed_hypotheses):
    for hypothesis in completed_hypotheses:
      hypothesis.score /= pow(len(hypothesis.id_list), self.m)

