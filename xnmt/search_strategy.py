import dynet as dy
import numpy as np
from collections import namedtuple

from xnmt.length_normalization import NoNormalization
from xnmt.serialize.serializer import bare
from xnmt.vocab import Vocab

# Output of the search
SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions'])

class SearchStrategy(object):
  '''
  A template class to generate translation from the output probability model.
  '''
  def generate_output(self, decoder, attender, output_embedder, dec_state, src_length=None, forced_trg_ids=None):
    """
    Args:
      decoder (Decoder):
      attender (Attender):
      output_embedder (Embedder): target embedder
      dec_state (MlpSoftmaxDecoderState): initial decoder state
      src_length (int): length of src sequence, required for some types of length normalization
      forced_trg_ids (List[int]): list of word ids, if given will force to generate this is the target sequence
    Returns:
      Tuple[List[int],float]: (id list, score)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(SearchStrategy):
  '''
  Performs greedy search (aka beam search with beam size 1)
  
  Args:
    max_len (int): maximum number of tokens to generate.
  '''
  def __init__(self, max_len=100):
    self.max_len = max_len
  def generate_output(self, decoder, attender, output_embedder, dec_state, src_length=None, forced_trg_ids=None):
    score = 0.0
    word_ids = []
    attentions = []

    while (word_ids==[] or word_ids[-1]!=Vocab.ES) and len(word_ids) < self.max_len:
      if len(word_ids) > 0: # don't feed in the initial start-of-sentence token
        dec_state = decoder.add_input(dec_state, output_embedder.embed(word_ids[-1] if forced_trg_ids is None else forced_trg_ids[len(word_ids)-1]))
      dec_state.context = attender.calc_context(dec_state.rnn_state.output())
      logsoftmax = dy.log_softmax(decoder.get_scores(dec_state)).npvalue()
      if forced_trg_ids is None:
        cur_id = np.argmax(logsoftmax)
      else:
        cur_id = forced_trg_ids[len(word_ids)]

      score += logsoftmax[cur_id]
      word_ids.append(cur_id)
      attentions.append(attender.get_last_attention())

    return SearchOutput(word_ids, attentions), score

class BeamSearch(SearchStrategy):
  
  """
  Performs beam search.
  
  Args:
    beam_size (int):
    max_len (int): maximum number of tokens to generate.
    len_norm (LengthNormalization): type of length normalization to apply
  """

  def __init__(self, beam_size, max_len=100, len_norm=bare(NoNormalization)):
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm

    self.entrs = []

  class Hypothesis:
    def __init__(self, score, output, state):
      self.score = score
      self.state = state
      self.output = output
    def __str__(self):
      return "hypo S=%s ids=%s" % (self.score, self.output.word_ids)
    def __repr__(self):
      return "hypo S=%s |ids|=%s" % (self.score, len(self.output.word_ids))

  def generate_output(self, decoder, attender, output_embedder, dec_state, src_length=None, forced_trg_ids=None):

    if forced_trg_ids is not None: assert self.beam_size == 1
    active_hyp = [self.Hypothesis(0, SearchOutput([], []), dec_state)]

    completed_hyp = []
    length = 0

    # TODO(philip30): Copying the output at each hypothesis expansion is not time efficient (memory efficient?).
    # every hyp should just store the output at its timestep, store the reference to the parent hyp
    # and do backtracking to collect all the outputs.
    while len(completed_hyp) < self.beam_size and length < self.max_len:
      new_set = []
      for hyp in active_hyp:
        dec_state = hyp.state
        if length > 0: # don't feed in the initial start-of-sentence token
          last_generated = hyp.output.word_ids[-1]
          if last_generated == Vocab.ES:
            completed_hyp.append(hyp)
            continue
          dec_state = decoder.add_input(dec_state, output_embedder.embed(last_generated if forced_trg_ids is None else forced_trg_ids[length-1]))
        dec_state.context = attender.calc_context(dec_state.rnn_state.output())
        score = dy.log_softmax(decoder.get_scores(dec_state)).npvalue()
        if forced_trg_ids is None:
          top_ids = np.argpartition(score, max(-len(score),-self.beam_size))[-self.beam_size:]
        else:
          top_ids = [forced_trg_ids[length]]

        for cur_id in top_ids:
          new_list = list(hyp.output.word_ids)
          new_list.append(cur_id)
          new_attn = list(hyp.output.attentions)
          new_attn.append(attender.get_last_attention())
          new_set.append(self.Hypothesis(self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list)),
                                         SearchOutput(new_list, new_attn),
                                         dec_state))
      length += 1

      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    self.len_norm.normalize_completed(completed_hyp, src_length)

    result = sorted(completed_hyp, key=lambda x: x.score, reverse=True)[0]
    return result.output, result.score
