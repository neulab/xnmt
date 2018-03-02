from __future__ import division, generators

import dynet as dy
import numpy as np

from xnmt.loss import LossBuilder
from xnmt.serialize.serializer import Serializable
from xnmt.vocab import Vocab
from xnmt.serialize.tree_tools import Ref, Path
import xnmt.evaluator
import xnmt.linear as linear


class LossCalculator(Serializable):
  '''
  A template class implementing the training strategy and corresponding loss calculation.
  '''
  yaml_tag = u'!LossCalculator'

  def __init__(self, loss_calculator = None):
    if loss_calculator is None:
      self.loss_calculator = MLELoss()
    else:
      self.loss_calculator = loss_calculator

  def __call__(self, translator, dec_state, src, trg):
      return self.loss_calculator(translator, dec_state, src, trg)


class MLELoss(Serializable):
  yaml_tag = '!MLELoss'

  def __call__(self, translator, dec_state, src, trg):
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    losses = []
    seq_len = len(trg[0]) if xnmt.batcher.is_batched(src) else len(trg)
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert len(single_trg) == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    for i in range(seq_len):
      ref_word = trg[i] if not xnmt.batcher.is_batched(src) \
                      else xnmt.batcher.mark_as_batch([single_trg[i] for single_trg in trg])

      dec_state.context = translator.attender.calc_context(dec_state.rnn_state.output())
      word_loss = translator.decoder.calc_loss(dec_state, ref_word)
      if xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(word_loss)
      if i < seq_len-1:
        dec_state = translator.decoder.add_input(dec_state, translator.trg_embedder.embed(ref_word))

    return dy.esum(losses)

def remove_eos(sequence, eos_sym=Vocab.ES):
  try:
    idx = sequence.index(Vocab.ES)
    sequence = sequence[:idx]
  except ValueError:
    # NO EOS
    pass
  return sequence

class ReinforceLoss(Serializable):
  yaml_tag = '!ReinforceLoss'

  def __init__(self, exp_global=Ref(Path("exp_global")), evaluation_metric=None, sample_length=50, use_baseline=False, decoder_hidden_dim=None):
    self.sample_length = sample_length
    self.use_baseline = use_baseline
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.BLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

    if self.use_baseline:
      model = exp_global.dynet_param_collection.param_col
      decoder_hidden_dim = decoder_hidden_dim or exp_global.default_layer_dim
      self.baseline = linear.Linear(input_dim=decoder_hidden_dim, output_dim=1, model=model)

  def __call__(self, model, dec_state, src, trg):
    # TODO: apply trg.mask ?
    logsofts, samples, hts = model.sample_one(dec_state, self.sample_length)
    # Calculate evaluation scores
    self.eval_score = []
    for trg_i, sample_i in zip(trg, samples):
      # Removing EOS
      sample_i = remove_eos(sample_i)
      ref_i = remove_eos(trg_i.words)
      # Evaluating 
      if len(sample_i) == 0:
        score = 0
      else:
        score = self.evaluation_metric.evaluate_fast(ref_i, sample_i)
      self.eval_score.append(score)
    self.true_score = dy.inputTensor(self.eval_score, batched=True)
    # Composing losses
    loss = LossBuilder()
    if self.use_baseline:
      baseline_loss = []
      score = []
      for i, (h_t, logsoft) in enumerate(zip(hts, logsofts)):
        bs_score = self.baseline(dy.nobackprop(h_t))
        baseline_loss.append(dy.squared_distance(self.true_score, bs_score))
        score.append(dy.cmult(logsoft, bs_score - self.true_score))
      loss.add_loss("reinforce", dy.sum_elems(dy.esum(score)))
      loss.add_loss("reinf_baseline", dy.sum_elems(dy.esum(baseline_loss)))
    else:
      loss.add_loss("reinforce", dy.sum_elems(dy.cmult(-self.true_score, dy.esum(logsofts))))
    return loss

class MinRiskLoss(Serializable):
  yaml_tag = '!MinRiskLoss'

  def __init__(self, exp_global=Ref(Path("exp_global")),
                     evaluation_metric=None,
                     sample_length=50,
                     sample_num=10,
                     alpha=5e-3):
    # Samples
    self.sample_length = sample_length
    self.sample_num = sample_num
    self.alpha = alpha
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.BLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

  # TODO Implement masking here!
  def __call__(self, model, dec_state, src, trg):
    batch_size = len(trg)
    samples = [set() for _ in range(batch_size)]
    deltas = []
    logprobs = []
    for i in range(self.sample_num):
      ref = trg if i == 0 else None
      logprob, sample, _ = model.sample_one(dec_state, self.sample_length, ref)
      logprob = dy.esum(logprob)
      # Calculate hash, and mask the same sample
      hashed = [hash(tuple(s)) for s in sample]
      flag = [1 if h not in s else 0 for (h, s) in zip(hashed, samples)]
      logprob = dy.cmult(dy.inputTensor(flag, batched=True), logprob) * self.alpha
      list(s.add(h) for s, h in zip(samples, hashed))
      # Calculate the evaluation score
      eval_score = [0 for _ in range(batch_size)]
      for j in range(batch_size):
        if flag[j] == 0: continue
        ref = remove_eos(trg[j].words)
        hyp = remove_eos(sample[j])
        if len(hyp) == 0:
          eval_score[j] = 0
        else:
          eval_score[j] = self.evaluation_metric.evaluate_fast(ref, hyp)
      deltas.append(-dy.inputTensor(eval_score, batched=True))
      logprobs.append(logprob)
    prob = dy.softmax(dy.exp(dy.concatenate(logprobs)))
    risk = dy.sum_elems(dy.cmult(prob, dy.concatenate(deltas)))
    return LossBuilder({"risk": risk})

