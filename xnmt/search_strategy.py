import dynet as dy
import numpy as np
from collections import namedtuple

from xnmt.length_normalization import *
from xnmt.vocab import Vocab

# Output of the search
SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions'])

class SearchStrategy(object):
  '''
  A template class to generate translation from the output probability model.
  '''
  def generate_output(self, decoder, attender, output_embedder, dec_state, src_length=None, forced_trg_ids=None):
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(SearchStrategy):
  '''
  Performs greedy search (aka beam search with beam size 1)
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

  def __init__(self, beam_size, max_len=100, len_norm=None):
    self.beam_size = beam_size
    self.max_len = max_len
    # The only reason why we don't set NoNormalization as the default is because it currently
    # breaks our documentation pipeline
    self.len_norm = len_norm if len_norm != None else NoNormalization()

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
    """
    :param decoder: decoder.Decoder subclass
    :param attender: attender.Attender subclass
    :param output_embedder: embedder.Embedder subclass
    :param dec_state: The decoder state
    :param src_length: length of src sequence, required for some types of length normalization
    :param forced_trg_ids: list of word ids, if given will force to generate this is the target sequence
    :returns: (id list, score)
    """

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

class TrDecBeamSearch(SearchStrategy):

  def __init__(self, beam_size, max_len=100, len_norm=None):
    self.beam_size = beam_size
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

    def __str__(self):
      return "hypo S=%s ids=%s" % (self.score, self.id_list)

    def __repr__(self):
      return "hypo S=%s |ids|=%s" % (self.score, len(self.id_list))

  def generate_output(self, decoder, attender, output_embedder, dec_state, src_length=None, word_embedder=None,
                      forced_trg_ids=None, trg_rule_vocab=None, tag_embedder=None, word_attender=None,
                      output_beam=False, word_vocab=None):
    """
    :param decoder: decoder.Decoder subclass
    :param attender: attender.Attender subclass
    :param output_embedder: embedder.Embedder subclass
    :param dec_state: The decoder state
    :param src_length: length of src sequence, required for some types of length normalization
    :param forced_trg_ids: list of word ids, if given will force to generate this is the target sequence
    :param trg_rule_vocab: RuleVocab object, if given decode with tree decoder
    :returns: (id list, score)
    """

    if forced_trg_ids is not None: assert self.beam_size == 1

    active_hyp = [self.Hypothesis(0, [], dec_state)]

    completed_hyp = []
    length = 0
    stop_action = False
    while len(completed_hyp) < self.beam_size and length < self.max_len:
      new_set = []
      for hyp in active_hyp:
        dec_state = hyp.state
        if length > 0:  # don't feed in the initial start-of-sentence token
          if trg_rule_vocab:
            if word_embedder:
              dec_state = decoder.add_input(dec_state,
                                            hyp.id_list[-1][0] if forced_trg_ids is None else forced_trg_ids[
                                              length - 1],
                                            word_embedder,
                                            output_embedder,
                                            word_vocab=word_vocab,
                                            trg_rule_vocab=trg_rule_vocab,
                                            tag_embedder=tag_embedder)
            else:
              dec_state = decoder.add_input(dec_state, output_embedder.embed(
                hyp.id_list[-1] if forced_trg_ids is None else forced_trg_ids[length - 1]),
                                            hyp.id_list[-1], trg_rule_vocab, tag_embedder)
            if len(dec_state.open_nonterms) == 0:
              # only know if the stack is empty after we add the current rule back to the tree
              completed_hyp.append(hyp)
              continue
          else:
            if hyp.id_list[-1] == Vocab.ES:
              completed_hyp.append(hyp)
              continue
            dec_state = decoder.add_input(dec_state, output_embedder.embed(
              hyp.id_list[-1] if forced_trg_ids is None else forced_trg_ids[length - 1]))
        dec_state.context = attender.calc_context(dec_state.rnn_state.output())

        if trg_rule_vocab:
          if decoder.set_word_lstm:
            dec_state.word_context = word_attender.calc_context(dec_state.word_rnn_state.output())
          # only keep rules with the correct rhs
          if word_embedder:
            is_terminal = dec_state.open_nonterms[-1].label == u'*'
            score, num_valid_rule, stop_prob, len_score = decoder.get_scores(dec_state, trg_rule_vocab,
                                                                             is_terminal=is_terminal)
          else:
            score, num_valid_rule = decoder.get_scores(dec_state, trg_rule_vocab)
          score = dy.log_softmax(score).npvalue()
        else:
          score = dy.log_softmax(decoder.get_scores(dec_state)).npvalue()
        if forced_trg_ids is None:
          if trg_rule_vocab and num_valid_rule >= 0:
            top_ids = np.argpartition(score, max(-num_valid_rule, -self.beam_size))[
                      -min(self.beam_size, num_valid_rule):]
          else:
            top_ids = np.argpartition(score, max(-len(score), -self.beam_size))[-self.beam_size:]
        else:
          top_ids = [forced_trg_ids[length]]

        for cur_id in top_ids:
          new_list = list(hyp.id_list)
          if word_embedder:
            # word/rule index, is_terminal, stop_action
            is_terminal = dec_state.open_nonterms[-1].label == u'*'
            stop_action = False
            #if is_terminal and word_vocab[cur_id].endswith(u'\u2581'): stop_action = True
            if is_terminal and not word_vocab[cur_id].endswith('@@'): stop_action = True
            new_list.append([cur_id, is_terminal, stop_action])
          else:
            new_list.append(cur_id)
          if trg_rule_vocab:
            if score[cur_id] == -np.inf: continue
            new_set.append(self.Hypothesis(
              #self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list), src_length=src_length),
              self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list)),
              new_list, dec_state.copy()))
          else:
            new_set.append(self.Hypothesis(
              #self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list), src_length=src_length),
              self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list)),
              new_list, dec_state))
      length += 1
      if trg_rule_vocab and num_valid_rule >= 0:
        active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:min(self.beam_size, num_valid_rule)]
      else:
        active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    self.len_norm.normalize_completed(completed_hyp, src_length)
    if completed_hyp:
      if output_beam:
        return [h.id_list for h in completed_hyp], [h.score for h in completed_hyp]
      else:
        result = sorted(completed_hyp, key=lambda x: x.score, reverse=True)[0]
        return result.id_list, result.score
    else:
      if word_embedder:
        return [[Vocab.ES, True, False]], [0]
      else:
        return [Vocab.ES], [0]