import dynet as dy
import numpy as np
from length_normalization import *

class SearchStrategy:
  '''
  A template class to generate translation from the output probability model.
  '''
  def generate_output(self):
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class BeamSearch(SearchStrategy):
  def __init__(self, b, decoder, attender, max_len=100, len_norm=None):
    self.b = b
    self.decoder = decoder
    self.attender = attender
    self.max_len = max_len
    if len_norm is None:
      self.len_norm = NoNormalization()
    else:
      self.len_norm = len_norm

  class Hypothesis:
    def __init__(self, score, id, state):
      self.score = score
      self.state = state
      self.id_list = id


  def generate_output(self):
    # TODO: Add suitable length normalization
    active_hypothesis = [self.Hypothesis(0, [0], self.decoder.state)]
    completed_hypothesis = []
    length = 0

    while len(completed_hypothesis) < self.b and length < self.max_len:
      length += 1
      new_set = []
      for hyp in active_hypothesis:

        if hyp.id_list[-1] == 1:
          completed_hypothesis.append(hyp)
          continue

        self.decoder.state = hyp.state
        self.decoder.add_input(hyp.id_list[-1])
        context = self.attender.calc_context(self.decoder.state.output())
        score = dy.log_softmax(self.decoder.get_scores(context)).npvalue()
        top_ids = np.argsort(score)[::-1][:self.b]

        for id in top_ids:
          new_list = list(hyp.id_list)
          new_list.append(id)
          new_set.append(self.Hypothesis(hyp.score + score[id], new_list, self.decoder.state))

      active_hypothesis = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.b]

    if len(completed_hypothesis) == 0:
      completed_hypothesis = active_hypothesis

    self.len_norm.normalize_length(completed_hypothesis)

    result = sorted(completed_hypothesis, key=lambda x: x.score, reverse=True)[0]
    return result.id_list