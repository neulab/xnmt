from typing import List, Optional, Sequence, Union
import numbers

import numpy as np

import xnmt
from xnmt import batchers, event_trigger, losses, search_strategies, sent, vocabs
from xnmt.persistence import bare, Ref, Serializable, serializable_init
from xnmt.modelparts import transforms
from xnmt.eval import metrics


class LossCalculator(object):
  """
  A template class implementing the training strategy and corresponding loss calculation.
  """
  def calc_loss(self,
                model: 'model_base.ConditionedModel',
                src: Union[sent.Sentence, 'batchers.Batch'],
                trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
    raise NotImplementedError()

  def remove_eos(self, sequence: Sequence[numbers.Integral], eos_sym=vocabs.Vocab.ES) -> Sequence[numbers.Integral]:
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
  """
  yaml_tag = '!MLELoss'
  @serializable_init
  def __init__(self) -> None:
    pass

  def calc_loss(self,
                model: 'model_base.ConditionedModel',
                src: Union[sent.Sentence, 'batchers.Batch'],
                trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
    loss = model.calc_nll(src, trg)
    return losses.FactoredLossExpr({"mle": loss})


class FeedbackLoss(Serializable, LossCalculator):
  """
  A loss that first calculates a standard loss function, then feeds it back to the
  model using the model.additional_loss function.

  Args:
    child_loss: The loss that will be fed back to the model
    repeat: Repeat the process multiple times and use the sum of the losses. This is
            useful when there is some non-determinism (such as sampling in the encoder, etc.)
  """
  yaml_tag = '!FeedbackLoss'
  @serializable_init
  def __init__(self,
               child_loss: LossCalculator = bare(MLELoss),
               repeat: numbers.Integral = 1) -> None:
    self.child_loss = child_loss
    self.repeat = repeat

  def calc_loss(self,
                model: 'model_base.ConditionedModel',
                src: Union[sent.Sentence, 'batcher.Batch'],
                trg: Union[sent.Sentence, 'batcher.Batch']) -> losses.FactoredLossExpr:
    loss_builder = losses.FactoredLossExpr()
    for _ in range(self.repeat):
      standard_loss = self.child_loss.calc_loss(model, src, trg)
      additional_loss = event_trigger.calc_additional_loss(trg, model, standard_loss)
      loss_builder.add_factored_loss_expr(standard_loss)
      loss_builder.add_factored_loss_expr(additional_loss)
    return loss_builder

if xnmt.backend_dynet:

  import dynet as dy

  class GlobalFertilityLoss(Serializable, LossCalculator):
    """
    A fertility loss according to Cohn+, 2016.
    Incorporating Structural Alignment Biases into an Attentional Neural Translation Model

    https://arxiv.org/pdf/1601.01085.pdf
    """
    yaml_tag = '!GlobalFertilityLoss'
    @serializable_init
    def __init__(self) -> None:
      pass

    def calc_loss(self,
                  model: 'model_base.ConditionedModel',
                  src: Union[sent.Sentence, 'batchers.Batch'],
                  trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
      assert hasattr(model, "attender") and hasattr(model.attender, "attention_vecs"), \
             "Must be called after MLELoss with models that have attender."
      masked_attn = model.attender.attention_vecs
      if trg.mask is not None:
        trg_mask = 1-(trg.mask.np_arr.transpose())
        masked_attn = [dy.cmult(attn, dy.inputTensor(mask, batched=True)) for attn, mask in zip(masked_attn, trg_mask)]

      loss = self.global_fertility(masked_attn)
      return losses.FactoredLossExpr({"global_fertility": loss})

    def global_fertility(self, a: Sequence[dy.Expression]) -> dy.Expression:
      return dy.sum_elems(dy.square(1 - dy.esum(a)))


  class CompositeLoss(Serializable, LossCalculator):
    """
    Summing losses from multiple LossCalculator.
    """
    yaml_tag = "!CompositeLoss"
    @serializable_init
    def __init__(self, pt_losses:List[LossCalculator], loss_weight: Optional[Sequence[numbers.Real]] = None) -> None:
      self.pt_losses = pt_losses
      if loss_weight is None:
        self.loss_weight = [1.0 for _ in range(len(pt_losses))]
      else:
        self.loss_weight = loss_weight
      assert len(self.loss_weight) == len(pt_losses)

    def calc_loss(self,
                  model: 'model_base.ConditionedModel',
                  src: Union[sent.Sentence, 'batchers.Batch'],
                  trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
      total_loss = losses.FactoredLossExpr()
      for loss, weight in zip(self.pt_losses, self.loss_weight):
        total_loss.add_factored_loss_expr(loss.calc_loss(model, src, trg) * weight)
      return total_loss


  class ReinforceLoss(Serializable, LossCalculator):
    """
    Reinforce Loss according to Ranzato+, 2015.
    SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS.

    (This is not the MIXER algorithm)

    https://arxiv.org/pdf/1511.06732.pdf
    """
    yaml_tag = '!ReinforceLoss'
    @serializable_init
    def __init__(self,
                 baseline:Optional[Serializable]=None,
                 evaluation_metric: metrics.SentenceLevelEvaluator = bare(metrics.FastBLEUEvaluator),
                 search_strategy: search_strategies.SearchStrategy = bare(search_strategies.SamplingSearch),
                 inv_eval: bool = True,
                 decoder_hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim")) -> None:
      self.inv_eval = inv_eval
      self.search_strategy = search_strategy
      self.evaluation_metric = evaluation_metric
      self.baseline = self.add_serializable_component("baseline", baseline,
                                                      lambda: transforms.Linear(input_dim=decoder_hidden_dim, output_dim=1))

    def calc_loss(self,
                  model: 'model_base.ConditionedModel',
                  src: Union[sent.Sentence, 'batchers.Batch'],
                  trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
      search_outputs = model.generate_search_output(src, self.search_strategy)
      sign = -1 if self.inv_eval else 1

      total_loss = losses.FactoredLossExpr()
      for search_output in search_outputs:
        # Calculate rewards
        eval_score = []
        for trg_i, sample_i in zip(trg, search_output.word_ids):
          # Removing EOS
          sample_i = self.remove_eos(sample_i.tolist())
          ref_i = trg_i.words[:trg_i.len_unpadded()]
          score = self.evaluation_metric.evaluate_one_sent(ref_i, sample_i)
          eval_score.append(sign * score)
        reward = dy.inputTensor(eval_score, batched=True)
        # Composing losses
        loss = losses.FactoredLossExpr()
        baseline_loss = []
        cur_losses = []
        for state, mask in zip(search_output.state, search_output.mask):
          bs_score = self.baseline.transform(dy.nobackprop(state.as_vector()))
          baseline_loss.append(dy.squared_distance(reward, bs_score))
          logsoft = model.decoder.scorer.calc_log_probs(state.as_vector())
          loss_i = dy.cmult(logsoft, reward - bs_score)
          cur_losses.append(dy.cmult(loss_i, dy.inputTensor(mask, batched=True)))
        loss.add_loss("reinforce", dy.sum_elems(dy.esum(cur_losses)))
        loss.add_loss("reinf_baseline", dy.sum_elems(dy.esum(baseline_loss)))
        # Total losses
        total_loss.add_factored_loss_expr(loss)
      return loss


  class MinRiskLoss(Serializable, LossCalculator):
    yaml_tag = '!MinRiskLoss'

    @serializable_init
    def __init__(self,
                 evaluation_metric: metrics.Evaluator = bare(metrics.FastBLEUEvaluator),
                 alpha: numbers.Real = 0.005,
                 inv_eval: bool = True,
                 unique_sample: bool = True,
                 search_strategy: search_strategies.SearchStrategy = bare(search_strategies.SamplingSearch)) -> None:
      # Samples
      self.alpha = alpha
      self.evaluation_metric = evaluation_metric
      self.inv_eval = inv_eval
      self.unique_sample = unique_sample
      self.search_strategy = search_strategy

    def calc_loss(self,
                  model: 'model_base.ConditionedModel',
                  src: Union[sent.Sentence, 'batchers.Batch'],
                  trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
      batch_size = trg.batch_size()
      uniques = [set() for _ in range(batch_size)]
      deltas = []
      probs = []
      sign = -1 if self.inv_eval else 1
      search_outputs = model.generate_search_output(src, self.search_strategy)
      for search_output in search_outputs:
        assert len(search_output.word_ids) == 1
        assert search_output.word_ids[0].shape == (len(search_output.state),)
        logprob = []
        for word, state in zip(search_output.word_ids[0], search_output.state):
          lpdist = model.decoder.scorer.calc_log_probs(state.as_vector())
          lp = dy.pick(lpdist, word)
          logprob.append(lp)
        sample = search_output.word_ids
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
              uniques[j].add(hash_val)
            # Calc evaluation score
          eval_score[j] = self.evaluation_metric.evaluate_one_sent(ref_j, hyp_j) * sign
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

      return losses.FactoredLossExpr({"risk": risk})


