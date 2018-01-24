from simple_settings import settings

import dynet as dy

from xnmt.serialize.serializer import Serializable
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.evaluator import LossScore
from xnmt.serialize.tree_tools import Path, Ref
import xnmt.xnmt_evaluate

class EvalTask:
  '''
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  '''
  def eval(self):
    raise NotImplementedError("EvalTask.eval needs to be implemented in child classes")

class LossEvalTask(Serializable):
  '''
  A task that does evaluation of the loss function.
  '''

  yaml_tag = u'!LossEvalTask'
  
  def __init__(self, src_file, ref_file, model=Ref(path=Path("model")),
                batcher=Ref(path=Path("train.batcher"), required=False),
                loss_calculator=None):
    self.model = model
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())
    self.src_file = src_file
    self.ref_file = ref_file
    self.batcher = batcher
    self.src_data = None

  def eval(self):
    if self.src_data == None:
      self.src_data, self.ref_data, self.src_batches, self.ref_batches = \
        xnmt.input.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                        self.src_file, self.ref_file, batcher=self.batcher)
    loss_val = 0
    ref_words_cnt = 0 
    for src, trg in zip(self.src_batches, self.ref_batches):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      standard_loss = dy.sum_batches(self.model.calc_loss(src, trg, self.loss_calculator))
      ref_words_cnt += self.model.trg_reader.count_words(trg)
      loss_val += standard_loss.value()
    return LossScore(loss_val / ref_words_cnt), ref_words_cnt

class AccuracyEvalTask(Serializable):
  '''
  A task that does evaluation of some measure of accuracy.
  '''

  yaml_tag = u'!AccuracyEvalTask'

  def __init__(self, src_file, ref_file, hyp_file, model=Ref(path=Path("model")),
               eval_metrics="bleu", inference=None, candidate_id_file=None):
    self.model = model
    self.eval_metrics = [s.lower() for s in eval_metrics.split(",")]
    self.src_file = src_file
    self.ref_file = ref_file
    self.hyp_file = hyp_file
    self.candidate_id_file = candidate_id_file
    self.inference = inference or self.model.inference
   
  def eval(self):
    self.inference(generator = self.model,
                   src_file = self.src_file,
                   trg_file = self.hyp_file,
                   candidate_id_file = self.candidate_id_file)
    # TODO: This is not ideal because it requires reading the data
    #       several times. Is there a better way?
    evaluate_args = {}
    evaluate_args["hyp_file"] = self.hyp_file
    evaluate_args["ref_file"] = self.ref_file
    # Evaluate
    eval_scores = []
    for eval_metric in self.eval_metrics:
      evaluate_args["evaluator"] = eval_metric
      eval_scores.append(xnmt.xnmt_evaluate.xnmt_evaluate(**evaluate_args))
    # Calculate the reference file size
    ref_words_cnt = 0
    for ref_sent in self.model.trg_reader.read_sents(self.ref_file):
      ref_words_cnt += self.model.trg_reader.count_words(ref_sent)
      ref_words_cnt += 0
    return eval_scores, ref_words_cnt

