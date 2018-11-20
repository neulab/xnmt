from collections import namedtuple
import math
from typing import Callable, List, Optional, Sequence
import numbers

import dynet as dy
import numpy as np

from xnmt import batchers, logger
from xnmt.modelparts import decoders
from xnmt.length_norm import NoNormalization, LengthNormalization
from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.vocabs import Vocab


SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions', 'score', 'logsoftmaxes', 'state', 'mask'])
"""
Output of the search
words_ids: list of generated word ids
attentions: list of corresponding attention vector of word_ids
score: a single value of log(p(E|F))
logsoftmaxes: a corresponding softmax vector of the score. score = logsoftmax[word_id]
state: a NON-BACKPROPAGATEABLE state that is used to produce the logsoftmax layer
       state is usually used to generate 'baseline' in reinforce loss
masks: whether the particular word id should be ignored or not (1 for not, 0 for yes)
"""


class SearchStrategy(object):
  """
  A template class to generate translation from the output probability model. (Non-batched operation)
  """
  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_state: decoders.AutoRegressiveDecoderState,
                      src_length: Optional[numbers.Integral] = None,
                      forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> List[SearchOutput]:
    """
    Args:
      translator: a translator
      initial_state: initial decoder state
      src_length: length of src sequence, required for some types of length normalization
      forced_trg_ids: list of word ids, if given will force to generate this is the target sequence
    Returns:
      List of (word_ids, attentions, score, logsoftmaxes)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(Serializable, SearchStrategy):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """

  yaml_tag = '!GreedySearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100) -> None:
    self.max_len = max_len

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_state: decoders.AutoRegressiveDecoderState,
                      src_length: Optional[numbers.Integral] = None,
                      forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> List[SearchOutput]:
    # Output variables
    score = []
    word_ids = []
    attentions = []
    logsoftmaxes = []
    states = []
    masks = []
    # Search Variables
    done = None
    current_state = initial_state
    for length in range(self.max_len):
      prev_word = word_ids[length-1] if length > 0 else None
      current_output = translator.generate_one_step(prev_word, current_state)
      current_state = current_output.state
      if forced_trg_ids is None:
        word_id = np.argmax(current_output.logsoftmax.npvalue(), axis=0)
        if len(word_id.shape) == 2:
          word_id = word_id[0]
      else:
        if batchers.is_batched(forced_trg_ids):
          word_id = [forced_trg_ids[i][length] for i in range(len(forced_trg_ids))]
        else:
          word_id = [forced_trg_ids[length]]
      logsoft = dy.pick_batch(current_output.logsoftmax, word_id)
      if done is not None:
        word_id = [word_id[i] if not done[i] else Vocab.ES for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        logsoft = dy.cmult(logsoft, dy.inputTensor(mask, batched=True))
        masks.append(mask)
      # Packing outputs
      score.append(logsoft.npvalue())
      word_ids.append(word_id)
      attentions.append(current_output.attention)
      logsoftmaxes.append(dy.pick_batch(current_output.logsoftmax, word_id))
      states.append(translator.get_nobp_state(current_state)) # TODO: not a part of translator interface, refactor
      # Check if we are done.
      done = [x == Vocab.ES for x in word_id]
      if all(done):
        break
    masks.insert(0, [1 for _ in range(len(done))])
    words = np.stack(word_ids, axis=1)
    score = np.sum(score, axis=0)
    return [SearchOutput(words, attentions, score, logsoftmaxes, states, masks)]

class BeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!BeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])
  
  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_state: decoders.AutoRegressiveDecoderState,
                      src_length: Optional[numbers.Integral] = None,
                      forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> List[SearchOutput]:
    # TODO(philip30): can only do single decoding, not batched
    assert forced_trg_ids is None or self.beam_size == 1
    if forced_trg_ids is not None and forced_trg_ids.sent_len() > self.max_len:
      logger.warning("Forced decoding with a target longer than max_len. "
                     "Increase max_len to avoid unexpected behavior.")

    active_hyp = [self.Hypothesis(0, None, None, None)]
    completed_hyp = []
    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        break
      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        if length > 0:
          prev_word = hyp.word
          prev_state = hyp.output.state
        else:
          prev_word = None
          prev_state = initial_state
        if prev_word == Vocab.ES:
          completed_hyp.append(hyp)
          continue
        current_output = translator.generate_one_step(prev_word, prev_state)
        score = current_output.logsoftmax.npvalue().transpose()
        if self.scores_proc:
          self.scores_proc(score)
        # Next Words
        if forced_trg_ids is None:
          top_words = np.argpartition(score, max(-len(score),-self.beam_size))[-self.beam_size:]
        else:
          top_words = [forced_trg_ids[length]]
        # Queue next states
        for cur_word in top_words:
          new_score = self.len_norm.normalize_partial_topk(hyp.score, score[cur_word], length + 1)
          new_set.append(self.Hypothesis(new_score, current_output, hyp, cur_word))
      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
    # There is no hyp reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp
    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)
    if self.one_best:
      hyp_and_score = [hyp_and_score[0]]
    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      logsoftmaxes = []
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        # TODO(philip30): This should probably be uncommented.
        # These 2 statements are an overhead because it is need only for reinforce and minrisk
        # Furthermore, the attentions is only needed for report.
        # We should have a global flag to indicate whether this is needed or not?
        # The global flag is modified if certain objects is instantiated.
        #logsoftmaxes.append(dy.pick(current.output.logsoftmax, current.word))
        #states.append(translator.get_nobp_state(current.output.state))
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(logsoftmaxes)),
                                  list(reversed(states)), None))
    return results

class SamplingSearch(Serializable, SearchStrategy):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol
  
  Args:
    max_len:
    sample_size:
  """

  yaml_tag = '!SamplingSearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100, sample_size: numbers.Integral = 5) -> None:
    self.max_len = max_len
    self.sample_size = sample_size

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_state: decoders.AutoRegressiveDecoderState,
                      src_length: Optional[numbers.Integral] = None,
                      forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> List[SearchOutput]:
    outputs = []
    for k in range(self.sample_size):
      if k == 0 and forced_trg_ids is not None:
        outputs.append(self.sample_one(translator, initial_state, forced_trg_ids))
      else:
        outputs.append(self.sample_one(translator, initial_state))
    return outputs
 
  # Words ids, attentions, score, logsoftmax, state
  def sample_one(self,
                 translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                 initial_state: decoders.AutoRegressiveDecoderState,
                 forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> SearchOutput:
    # Search variables
    current_words = None
    current_state = initial_state
    done = None
    # Outputs
    logsofts = []
    samples = []
    states = []
    attentions = []
    masks = []
    # Sample to the max length
    for length in range(self.max_len):
      translator_output = translator.generate_one_step(current_words, current_state)
      if forced_trg_ids is None:
        sample = translator_output.logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()
        if len(sample.shape) == 2:
          sample = sample[0]
      else:
        sample = [forced_trg[length] if forced_trg.sent_len() > length else Vocab.ES for forced_trg in forced_trg_ids]
      logsoft = dy.pick_batch(translator_output.logsoftmax, sample)
      if done is not None:
        sample = [sample[i] if not done[i] else Vocab.ES for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        logsoft = dy.cmult(logsoft, dy.inputTensor(mask, batched=True))
        masks.append(mask)
      # Appending output
      logsofts.append(logsoft)
      samples.append(sample)
      states.append(translator.get_nobp_state(translator_output.state))
      attentions.append(translator_output.attention)
      # Next time step
      current_words = sample
      current_state = translator_output.state
      # Check done
      done = [x == Vocab.ES for x in sample]
      # Check if we are done.
      if all(done):
        break
    # Packing output
    scores = dy.esum(logsofts).npvalue()
    masks.insert(0, [1 for _ in range(len(done))])
    samples = np.stack(samples, axis=1)
    return SearchOutput(samples, attentions, scores, logsofts, states, masks)


class MctsNode(object):
  def __init__(self,
               parent: Optional['MctsNode'],
               prior_dist: np.ndarray,
               word: Optional[numbers.Integral],
               attention: Optional[List[np.ndarray]],
               translator: 'xnmt.models.translators.AutoRegressiveTranslator',
               dec_state: decoders.AutoRegressiveDecoderState) -> None:
    self.parent = parent
    self.prior_dist = prior_dist  # log of softmax
    self.word = word
    self.attention = attention

    self.translator = translator
    self.dec_state = dec_state

    self.tries = 0
    self.avg_value = 0.0
    self.children = {}

    # If the child is unvisited, set its avg_value to
    # parent value - reduction where reduction = c * sqrt(sum of scores of all visited children)
    # where c is 0.25 in leela
    self.reduction = 0.0

  def choose_child(self) -> numbers.Integral:
    return max(range(len(self.prior_dist)), key=lambda move: self.compute_priority(move))

  def compute_priority(self, move: numbers.Integral) -> numbers.Real:
    if move not in self.children:
      child_val = self.prior_dist[move] + self.avg_value - self.reduction
      child_tries = 0
    else:
      child_val = self.prior_dist[move] + self.children[move].avg_value
      child_tries = self.children[move].tries

    K = 5.0
    exp_term = math.sqrt(1.0 * self.tries + 1.0) / (child_tries + 1)
    # TODO: This exp could be done before the prior is passed into the MctsNode
    # so it's done as a big batch
    exp_term *= K * math.exp(self.prior_dist[move])
    total_value = child_val + exp_term
    return total_value

  def expand(self) -> 'MctsNode':
    if self.word == Vocab.ES:
      return self

    move = self.choose_child()
    if move in self.children:
      return self.children[move].expand()
    else:
      output = self.translator.generate_one_step(move, self.dec_state)
      prior_dist = output.logsoftmax.npvalue()
      attention = output.attention

      path = []
      node = self
      while node is not None:
        path.append(node.word)
        node = node.parent
      path = ' '.join(str(word) for word in reversed(path))
      print('Creating new node:', path, '+', move)
      new_node = MctsNode(self, prior_dist, move, attention,
                          self.translator, output.state)
      self.children[move] = new_node
      return new_node

  def rollout(self, sample_func, max_len):
    prefix = []
    scores = []
    prev_word = None
    dec_state = self.dec_state

    if self.word == Vocab.ES:
      return prefix, scores

    while True:
      output = self.translator.generate_one_step(prev_word, dec_state)
      logsoftmax = output.logsoftmax.npvalue()
      attention = output.attention
      best_id = sample_func(logsoftmax)
      print("Rolling out node with word=", best_id, 'score=', logsoftmax[best_id])

      prefix.append(best_id)
      scores.append(logsoftmax[best_id])

      if best_id == Vocab.ES or len(prefix) >= max_len:
        break
      prev_word = best_id
      dec_state = output.state
    return prefix, scores

  def backup(self, result):
    print('Backing up', result)
    self.avg_value = self.avg_value * (self.tries / (self.tries + 1)) + result / (self.tries + 1)
    self.tries += 1
    if self.parent is not None:
      my_prob = self.parent.prior_dist[self.word]
      self.parent.backup(result + my_prob)

  def collect(self, words, attentions):
    if self.word is not None:
      words.append(self.word)
      attentions.append(self.attention)
    if len(self.children) > 0:
      best_child = max(self.children.itervalues(), key=lambda child: child.visits)
      best_child.collect(words, attentions)


def random_choice(logsoftmax: np.ndarray) -> numbers.Integral:
  #logsoftmax *= 100
  probs = np.exp(logsoftmax)
  probs /= sum(probs)
  choices = np.random.choice(len(probs), 1, p=probs)
  return choices[0]


def greedy_choice(logsoftmax: np.ndarray) -> numbers.Integral:
  return np.argmax(logsoftmax)


class MctsSearch(Serializable, SearchStrategy):
  """
  Performs search with Monte Carlo Tree Search
  """
  yaml_tag = '!MctsSearch'

  @serializable_init
  def __init__(self, visits: numbers.Integral = 200, max_len: numbers. Integral = 100) -> None:
    self.max_len = max_len
    self.visits = visits

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      dec_state: decoders.AutoRegressiveDecoderState,
                      src_length: Optional[numbers.Integral] = None,
                      forced_trg_ids: Optional[Sequence[numbers.Integral]] = None) -> List[SearchOutput]:
    assert forced_trg_ids is None
    orig_dec_state = dec_state

    output = translator.generate_one_step(None, dec_state)
    dec_state = output.state
    assert dec_state == orig_dec_state
    logsoftmax = output.logsoftmax.npvalue()
    root_node = MctsNode(None, logsoftmax, None, None, translator, dec_state)
    for i in range(self.visits):
      terminal = root_node.expand()
      words, scores = terminal.rollout(random_choice, self.max_len)
      terminal.backup(sum(scores))
      print()

    print('Final stats:')
    for word in root_node.children:
      print (word, root_node.compute_priority(word), root_node.prior_dist[word] + root_node.children[word].avg_value, root_node.children[word].tries)
    print()

    scores = []
    logsoftmaxes = []
    word_ids = []
    attentions = []
    states = []
    masks = []

    node = root_node
    while True:
      if len(node.children) == 0:
        break
      best_word = max(node.children, key=lambda word: node.children[word].tries)
      score = node.prior_dist[best_word]
      attention = node.children[best_word].attention

      scores.append(score)
      logsoftmaxes.append(node.prior_dist)
      word_ids.append(best_word)
      attentions.append(attention)
      states.append(node.dec_state)
      masks.append(1)

      node = node.children[best_word]

    word_ids = np.expand_dims(word_ids, axis=0)
    return [SearchOutput(word_ids, attentions, scores, logsoftmaxes, states, masks)]
