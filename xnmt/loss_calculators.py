from typing import Union

import dynet as dy
import numpy as np

from xnmt.losses import FactoredLossExpr
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.vocabs import Vocab
from xnmt.transforms import Linear
from xnmt import batchers, eval_metrics
import xnmt.input

class LossCalculator(object):
  """
  A template class implementing the training strategy and corresponding loss calculation.
  """
  def calc_loss(self, translator, src, trg):
    raise NotImplementedError()

  def remove_eos(self, sequence, eos_sym=Vocab.ES):
    try:
      idx = sequence.index(eos_sym)
      sequence = sequence[:idx]
    except ValueError:
      # NO EOS
      pass
    return sequence

class MLELoss(Serializable, LossCalculator):
  """
  Max likelihood loss calculator.

  Args:
    repeat: The calculation of maximum likelihood loss will be repeated this many times.
            This is only helpful when there is some inherent non-determinism in the
            training process (e.g. an encoder that uses sampling)
  """
  yaml_tag = '!MLELoss'
  @serializable_init
  def __init__(self,
               repeat: int = 1) -> None:
    self.repeat = repeat

  def calc_loss(self,
                model: 'model_base.ConditionedModel',
                src: Union[xnmt.input.Input, 'batcher.Batch'],
                trg: Union[xnmt.input.Input, 'batcher.Batch']):
    losses = [model.calc_nll(src, trg) for _ in range(self.repeat)]
    return FactoredLossExpr({"mle": losses[0] if self.repeat == 1 else dy.esum(losses)})

class ReinforceLoss(Serializable, LossCalculator):
  yaml_tag = '!ReinforceLoss'

  # TODO: document me
  @serializable_init
  def __init__(self, evaluation_metric=None, sample_length=50, use_baseline=False,
               inv_eval=True, decoder_hidden_dim=Ref("exp_global.default_layer_dim"), baseline=None):
    self.use_baseline = use_baseline
    self.inv_eval = inv_eval
    if evaluation_metric is None:
      self.evaluation_metric = eval_metrics.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

    if self.use_baseline:
      self.baseline = self.add_serializable_component("baseline", baseline,
                                                      lambda: Linear(input_dim=decoder_hidden_dim, output_dim=1))

  def calc_loss(self, translator, src, trg):
    # TODO(philip30): currently only using the best hypothesis / first sample for reinforce loss
    # A small further implementation is needed if we want to do reinforce with multiple samples.
    search_output = translator.generate_search_output(src, translator.search_strategy)[0] 
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
      self.evaluation_metric = eval_metrics.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric
    self.inv_eval = inv_eval
    self.unique_sample = unique_sample

  def calc_loss(self, translator, src, trg):
    batch_size = trg.batch_size()
    uniques = [set() for _ in range(batch_size)]
    deltas = []
    probs = []
    
    search_outputs = translator.generate_search_output(src, translator.search_strategy) 
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
            mask[j] = -1e20 # represents negative infinity
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

