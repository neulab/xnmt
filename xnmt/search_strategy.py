import dynet as dy
import numpy as np
from length_normalization import *
from vocab import Vocab

class SearchStrategy:
  '''
  A template class to generate translation from the output probability model.
  '''
  def generate_output(self):
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class BeamSearch(SearchStrategy):

  def __init__(self, b, max_len=100, len_norm=None):
    self.b = b
    self.max_len = max_len
    # The only reason why we don't set NoNormalization as the default is because it currently
    # breaks our documentation pipeline
    self.len_norm = len_norm if len_norm != None else NoNormalization()
    
    self.entrs = []

  class Hypothesis:
    def __init__(self, score, id_list, state):
      self.score = score
      self.state = state
      self.id_list = id_list



  def generate_output(self, decoder, attender, output_embedder, src_length=0):
    active_hyp = [self.Hypothesis(0, [0], decoder.state)]

    completed_hyp = []
    length = 0

    while len(completed_hyp) < self.b and length < self.max_len:
      length += 1
      new_set = []
      for hyp in active_hyp:

        if hyp.id_list[-1] == Vocab.ES:
          completed_hyp.append(hyp)
          continue


        decoder.state = hyp.state
        decoder.add_input(output_embedder.embed(hyp.id_list[-1]))
        context = attender.calc_context(decoder.state.output())
        score = dy.log_softmax(decoder.get_scores(context)).npvalue()
        top_ids = np.argpartition(score, max(-len(score),-self.b))[-self.b:]

        for cur_id in top_ids:
          new_list = list(hyp.id_list)
          new_list.append(cur_id)
          new_set.append(self.Hypothesis(self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list)), 
                                         new_list, 
                                         decoder.state))

      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.b]

    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    self.len_norm.normalize_completed(completed_hyp, src_length)

    result = sorted(completed_hyp, key=lambda x: x.score, reverse=True)[0]
    return result.id_list
