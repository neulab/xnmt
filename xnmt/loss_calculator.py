from typing import Union

import dynet as dy
import numpy as np

from xnmt.loss import FactoredLossExpr
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.vocab import Vocab
from xnmt.constants import INFINITY
from xnmt.transform import Linear
import xnmt.evaluator
import xnmt.batcher

class LossCalculator(object):
  """
  A template class implementing the training strategy and corresponding loss calculation.
  """
  def calc_loss(self, translator, initial_state, src, trg):
    raise NotImplementedError()

  def remove_eos(self, sequence, eos_sym=Vocab.ES):
    try:
      idx = sequence.index(eos_sym)
      sequence = sequence[:idx]
    except ValueError:
      # NO EOS
      pass
    return sequence

class AutoRegressiveMLELoss(Serializable, LossCalculator):
  """
  Max likelihood loss calculator for autoregressive models.

  Args:
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """
  yaml_tag = '!AutoRegressiveMLELoss'
  @serializable_init
  def __init__(self, truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.truncate_dec_batches = truncate_dec_batches

  def calc_loss(self, translator: 'translator.AutoRegressiveTranslator',
                initial_state: 'translator.AutoRegressiveDecoderState',
                src: Union[xnmt.input.Input, 'batcher.Batch'],
                trg: Union[xnmt.input.Input, 'batcher.Batch']):
    dec_state = initial_state
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    losses = []
    seq_len = trg.sent_len()
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    input_word = None
    for i in range(seq_len):
      ref_word = AutoRegressiveMLELoss._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)
      if self.truncate_dec_batches and xnmt.batcher.is_batched(ref_word):
        dec_state.rnn_state, ref_word = xnmt.batcher.truncate_batches(dec_state.rnn_state, ref_word)
      dec_state, word_loss = translator.calc_loss_one_step(dec_state, ref_word, input_word)
      if not self.truncate_dec_batches and xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(word_loss)
      input_word = ref_word

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in losses])
    else:
      loss_expr = dy.esum(losses)
    return FactoredLossExpr({"mle": loss_expr})

  @staticmethod
  def _select_ref_words(sent, index, truncate_masked = False):
    if truncate_masked:
      mask = sent.mask if xnmt.batcher.is_batched(sent) else None
      if not xnmt.batcher.is_batched(sent):
        return sent[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sent):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return xnmt.batcher.mark_as_batch(ret)
    else:
      if not xnmt.batcher.is_batched(sent): return sent[index]
      else: return xnmt.batcher.mark_as_batch([single_trg[index] for single_trg in sent])

class ReinforceLoss(Serializable, LossCalculator):
  yaml_tag = '!ReinforceLoss'

  # TODO: document me
  @serializable_init
  def __init__(self, evaluation_metric=None, sample_length=50, use_baseline=False,
               inv_eval=True, decoder_hidden_dim=Ref("exp_global.default_layer_dim"), baseline=None):
    self.use_baseline = use_baseline
    self.inv_eval = inv_eval
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

    if self.use_baseline:
      self.baseline = self.add_serializable_component("baseline", baseline,
                                                      lambda: Linear(input_dim=decoder_hidden_dim, output_dim=1))

  def calc_loss(self, translator, initial_state, src, trg):
    # TODO(philip30): currently only using the best hypothesis / first sample for reinforce loss
    # A small further implementation is needed if we want to do reinforce with multiple samples.
    search_output = translator.search_strategy.generate_output(translator, initial_state)[0]
    # Calculate evaluation scores
    self.eval_score = []
    for trg_i, sample_i in zip(trg, search_output.word_ids):
      # Removing EOS
      sample_i = self.remove_eos(sample_i.tolist())
      ref_i = self.remove_eos(trg_i.words)
      # Evaluating 
      if len(sample_i) == 0:
        score = 0
      else:
        score = self.evaluation_metric.evaluate(ref_i, sample_i) * \
                (-1 if self.inv_eval else 1)
      self.eval_score.append(score)
    self.true_score = dy.inputTensor(self.eval_score, batched=True)
    # Composing losses
    loss = FactoredLossExpr()
    if self.use_baseline:
      baseline_loss = []
      losses = []
      for state, logsoft, mask in zip(search_output.state,
                                      search_output.logsoftmaxes,
                                      search_output.mask):
        bs_score = self.baseline(state)
        baseline_loss.append(dy.squared_distance(self.true_score, bs_score))
        loss_i = dy.cmult(logsoft, self.true_score - bs_score)
        losses.append(dy.cmult(loss_i, dy.inputTensor(mask, batched=True)))
      loss.add_loss("reinforce", dy.sum_elems(dy.esum(losses)))
      loss.add_loss("reinf_baseline", dy.sum_elems(dy.esum(baseline_loss)))
    else:
      loss.add_loss("reinforce", dy.sum_elems(dy.cmult(self.true_score, dy.esum(logsofts))))
    return loss

class MinRiskLoss(Serializable, LossCalculator):
  yaml_tag = '!MinRiskLoss'

  @serializable_init
  def __init__(self, evaluation_metric=None, alpha=0.005, inv_eval=True, unique_sample=True):
    # Samples
    self.alpha = alpha
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric
    self.inv_eval = inv_eval
    self.unique_sample = unique_sample

  def calc_loss(self, translator, initial_state, src, trg):
    batch_size = trg.batch_size()
    uniques = [set() for _ in range(batch_size)]
    deltas = []
    probs = []
    
    search_outputs = translator.search_strategy.generate_output(translator, initial_state, forced_trg_ids=trg)
    for search_output in search_outputs:
      logprob = search_output.logsoftmaxes
      sample = search_output.word_ids
      attentions = search_output.attentions

      logprob = dy.esum(logprob) * self.alpha
      # Calculate the evaluation score
      eval_score = np.zeros(batch_size, dtype=float)
      mask = np.zeros(batch_size, dtype=float)
      for j in range(batch_size):
        ref_j = self.remove_eos(trg[j].words)
        hyp_j = self.remove_eos(sample[j].tolist())
        if self.unique_sample:
          hash_val = hash(tuple(hyp_j))
          if len(hyp_j) == 0 or hash_val in uniques[j]:
            mask[j] = -INFINITY
            continue
          else:
            # Count this sample in
            uniques[j].add(hash_val)
          # Calc evaluation score
        eval_score[j] = self.evaluation_metric.evaluate(ref_j, hyp_j) * \
                        (-1 if self.inv_eval else 1)
      # Appending the delta and logprob of this sample
      prob = logprob + dy.inputTensor(mask, batched=True)
      deltas.append(dy.inputTensor(eval_score, batched=True))
      probs.append(prob)
    sample_prob = dy.softmax(dy.concatenate(probs))
    deltas = dy.concatenate(deltas)
    risk = dy.sum_elems(dy.cmult(sample_prob, deltas))

    ### Debug
    #print(sample_prob.npvalue().transpose()[0])
    #print(deltas.npvalue().transpose()[0])
    #print("----------------------")
    ### End debug

    return FactoredLossExpr({"risk": risk})

