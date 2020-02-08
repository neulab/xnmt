import numpy as np
import numbers

from typing import List

import xnmt.search_strategies as search_strategies

import xnmt

from xnmt.vocabs import Vocab
from xnmt.persistence import serializable_init, Serializable
from xnmt.events import handle_xnmt_event, register_xnmt_handler
from xnmt.simultaneous.simult_translators import SimultaneousTranslator
from xnmt.simultaneous.simult_state import SimultaneousState

@xnmt.require_dynet
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
                      translator: SimultaneousTranslator,
                      initial_state: SimultaneousState,
                      src_length=None) -> List[search_strategies.SearchOutput]:
    # Output variables
    scores = []
    word_ids = []
    attentions = []
    logsoftmaxes = []
    # Search Variables
    current_state = initial_state

    encoding = []
    while current_state.has_been_written < self.max_len:
      action = translator.next_action(current_state, self.src_sent.sent_len(), len(encoding))
      if action == translator.Action.READ:
        # Reading
        current_state = current_state.read(self.src_sent)
        encoding.append(current_state.encoder_state.output())
      else:
        # Writing
        current_state = current_state.calc_context(encoding)
        current_output = translator.add_input(current_state.prev_written_word, current_state)
        best_words, best_scores = translator.best_k(current_output.state, 1)
        current_state = current_state.write(best_words[0])
        # Scoring
        scores.append(best_scores[0])
        word_ids.append(best_words[0])
        attentions.append(current_output.attention)
        current_word = best_words[0]
        if current_word == Vocab.ES:
          break

    score = np.sum(scores, axis=0)
    return [search_strategies.SearchOutput([word_ids], attentions, [score], [], [])]

