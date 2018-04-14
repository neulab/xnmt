import dynet as dy
import numpy as np
from collections import namedtuple

from xnmt.length_normalization import NoNormalization
from xnmt.persistence import bare, Serializable, serializable_init, Ref, bare
from xnmt.vocab import Vocab


# Output of the search
# words_ids: list of generated word ids
# attentions: list of corresponding attention vector of word_ids
# score: a single value of log(p(E|F))
# logsoftmaxes: a corresponding softmax vector of the score. score = logsoftmax[word_id]
# state: a NON-BACKPROPAGATEABLE state that is used to produce the logsoftmax layer
#        state is usually used to generate 'baseline' in reinforce loss
SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions', 'score', 'logsoftmaxes', 'state', 'mask'])

class SearchStrategy(object):
  '''
  A template class to generate translation from the output probability model. (Non-batched operation)
  '''
  def generate_output(self, translator, dec_state,
                      src_length=None, forced_trg_ids=None):
    """
    Args:
      translator (Translator): a translator
      dec_state (MlpSoftmaxDecoderState): initial decoder state
      src_length (int): length of src sequence, required for some types of length normalization
      forced_trg_ids (List[int]): list of word ids, if given will force to generate this is the target sequence
    Returns:
      List[SearchOutput]: List of (word_ids, attentions, score, logsoftmaxes)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(Serializable, SearchStrategy):
  '''
  Performs greedy search (aka beam search with beam size 1)
  
  Args:
    max_len (int): maximum number of tokens to generate.
  '''

  yaml_tag = '!GreedySearch'

  @serializable_init
  def __init__(self, max_len=100):
    self.max_len = max_len

  def generate_output(self, translator, initial_state,
                      src_length=None, forced_trg_ids=None):
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
    current_output = None
    for length in range(self.max_len):
      prev_word = word_ids[length-1] if length > 0 else None
      current_output = translator.output_one_step(prev_word, current_state)
      current_state = current_output.state
      if forced_trg_ids is None:
        word_id = np.argmax(current_output.logsoftmax.npvalue(), axis=0)
        if len(word_id.shape) == 2:
          word_id = word_id[0]
      else:
        word_id = [forced_trg_ids[i][length] for i in range(len(forced_trg_ids))]
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
      states.append(dy.nobackprop(current_state.rnn_state.output()))
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
    beam_size (int):
    max_len (int): maximum number of tokens to generate.
    len_norm (LengthNormalization): type of length normalization to apply
    one_best (bool): Whether to output the best hyp only or all completed hyps.
  """

  yaml_tag = '!BeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])
  
  @serializable_init
  def __init__(self, beam_size=3, max_len=100, len_norm=bare(NoNormalization), one_best=True):
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.entrs = []
    self.one_best = one_best

  def generate_output(self, translator, initial_state,
                      src_length=None, forced_trg_ids=None):
    # TODO(philip30): can only do single decoding, not batched
    assert forced_trg_ids is None or self.beam_size == 1
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
        current_output = translator.output_one_step(prev_word, prev_state)
        score = current_output.logsoftmax.npvalue().transpose()[0]
        # Next Words
        if forced_trg_ids is None:
          top_words = np.argpartition(score, max(-len(score),-self.beam_size))[-self.beam_size:]
        else:
          top_words = [forced_trg_ids[length]]
        # Queue next states
        for cur_word in top_words:
          new_score = self.len_norm.normalize_partial(hyp.score, score[cur_word], length+1)
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
      while current.parent != None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        logsoftmaxes.append(dy.pick(current.output.logsoftmax, current.word))
        states.append(dy.nobackprop(current.output.state.rnn_state.output()))
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(logsoftmaxes)), [states], None))
    return results

class SamplingSearch(Serializable, SearchStrategy):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol
  
  Args:
    max_length (int):
    sample_size (int): 
  """

  yaml_tag = '!SamplingSearch'

  @serializable_init
  def __init__(self, max_length=100, sample_size=5):
    self.max_length = max_length
    self.sample_size = sample_size

  def generate_output(self, translator, initial_state,
                      src_length=None, forced_trg_ids=None):
    outputs = []
    for k in range(self.sample_size):
      if k == 0 and forced_trg_ids is not None:
        outputs.append(self.sample_one(translator, initial_state, forced_trg_ids))
      else:
        outputs.append(self.sample_one(translator, initial_state))
    return outputs
 
  # Words ids, attentions, score, logsoftmax, state
  def sample_one(self, translator, initial_state, forced_trg_ids=None):
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
    for length in range(self.max_length):
      translator_output = translator.output_one_step(current_words, current_state)
      if forced_trg_ids is None:
        sample = translator_output.logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()
        if len(sample.shape) == 2:
          sample = sample[0]
      else:
        sample = [forced_trg[length] if len(forced_trg) > length else Vocab.ES for forced_trg in forced_trg_ids]
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
      states.append(dy.nobackprop(translator_output.state.rnn_state.output()))
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

