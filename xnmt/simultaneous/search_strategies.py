import numpy as np
import dynet as dy
import numbers

from typing import List

import xnmt.modelparts.decoders as decoders
import xnmt.search_strategies as search_strategies
import xnmt.event_trigger as event_trigger

from xnmt.vocabs import Vocab
from xnmt.persistence import serializable_init, Serializable
from xnmt.events import handle_xnmt_event, register_xnmt_handler

class SimultaneousGreedySearch(search_strategies.SearchStrategy, Serializable):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """

  yaml_tag = '!SimultaneousGreedySearch'

  @serializable_init
  @register_xnmt_handler
  def __init__(self, max_len: numbers.Integral = 100) -> None:
    self.max_len = max_len
    self.src_sent = None

  @handle_xnmt_event
  def on_start_sent(self, src_sent):
    self.src_sent = src_sent[0]
  
  def generate_output(self,
                      translator,
                      initial_state,
                      src_length=None,
                      forced_trg_ids=None) -> List[search_strategies.SearchOutput]:
    # Output variables
    scores = []
    word_ids = []
    attentions = []
    logsoftmaxes = []
    # Search Variables
    current_state = initial_state
    
    encoding = []
    next_word = None
    while current_state.to_write < self.max_len:
      action = translator.next_action(current_state, self.src_sent.sent_len(), len(encoding))
      if action == translator.Action.READ:
        # Reading
        current_state = current_state.read(self.src_sent)
        encoding.append(current_state.encoder_state.output())
      else:
        # Writing
        current_state = current_state.calc_context(encoding, next_word)
        current_output = translator.generate_one_step(None, current_state)
        next_word = np.argmax(current_output.logsoftmax.npvalue(), axis=0)
        if len(next_word.shape) == 2:
          next_word = next_word[0]
        current_state = current_state.write(next_word)
        # Scoring
        scores.append(dy.pick(current_output.logsoftmax, next_word))
        word_ids.append(next_word)
        attentions.append(current_output.attention)
        logsoftmaxes.append(current_output.logsoftmax)
        if next_word == Vocab.ES:
          break
    
    score = np.sum(scores, axis=0)
    return [search_strategies.SearchOutput([word_ids], attentions, [score], logsoftmaxes, [], [])]


