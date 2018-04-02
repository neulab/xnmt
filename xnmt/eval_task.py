from simple_settings import settings

import dynet as dy

import xnmt.input_reader
from xnmt.serialize.serializer import Serializable
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.evaluator import LossScore
from xnmt.serialize.tree_tools import Path, Ref
from xnmt.loss import LossBuilder, LossScalarBuilder
import xnmt.xnmt_evaluate
from xnmt.input import TreeReader

class EvalTask(object):
  '''
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  '''
  def eval(self):
    raise NotImplementedError("EvalTask.eval needs to be implemented in child classes")

class LossEvalTask(Serializable):
  '''
  A task that does evaluation of the loss function.

  Args:
    src_file (str):
    ref_file (str):
    model (GeneratorModel):
    batcher (Batcher):
    loss_calculator (LossCalculator):
    max_src_len (int):
    max_trg_len (int):
    desc (str):
  '''

  yaml_tag = '!LossEvalTask'

  def __init__(self, src_file, ref_file, model=Ref(path=Path("model")),
                batcher=Ref(path=Path("train.batcher"), required=False),
                loss_calculator=None, max_src_len=None, max_trg_len=None,
                desc=None, ref_len_file=None):
    self.model = model
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())
    self.src_file = src_file
    self.ref_file = ref_file
    self.batcher = batcher
    self.src_data = None
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.desc=desc
    self.ref_len_file = ref_len_file

  def eval(self):
    if self.src_data == None:
      self.src_data, self.ref_data, self.src_batches, self.ref_batches, self.ref_len_batches = \
        xnmt.input.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                        self.src_file, self.ref_file, ref_len_file=self.ref_len_file,
                                        batcher=self.batcher,
                                        max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
    loss_val = LossScalarBuilder()
    ref_words_cnt = 0
    i = 0
    for src, trg in zip(self.src_batches, self.ref_batches):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)

      loss_builder = LossBuilder()
      standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
      additional_loss = self.model.calc_additional_loss(standard_loss)
      loss_builder.add_loss("standard_loss", standard_loss)
      loss_builder.add_loss("additional_loss", additional_loss)

      if self.ref_len_batches is not None:
        ref_words_cnt += sum(self.ref_len_batches[i])
      else:
        ref_words_cnt += self.model.trg_reader.count_words(trg)
      loss_val += loss_builder.get_loss_stats()
      i += 1
    loss_stats = {k: v/ref_words_cnt for k, v in loss_val.items()}

    try:
      return LossScore(loss_stats[self.model.get_primary_loss()], loss_stats=loss_stats, desc=self.desc), ref_words_cnt
    except KeyError:
      raise RuntimeError("Did you wrap your loss calculation with LossBuilder({'primary_loss': loss_value}) ?")

class AccuracyEvalTask(Serializable):
  '''
  A task that does evaluation of some measure of accuracy.

  Args:
    src_file (str):
    ref_file (str):
    hyp_file (str):
    model (GeneratorModel):
    eval_metrics (str): comma-separated list of evaluation metrics
    inference (SimpleInference):
    candidate_id_file (str):
    desc (str):
  '''

  yaml_tag = '!AccuracyEvalTask'

  def __init__(self, src_file, ref_file, hyp_file, model=Ref(path=Path("model")),
               eval_metrics="bleu", inference=None, candidate_id_file=None,
               desc=None):
    self.model = model
    self.eval_metrics = [s.lower() for s in eval_metrics.split(",")]
    self.src_file = src_file
    self.ref_file = ref_file
    self.hyp_file = hyp_file
    self.candidate_id_file = candidate_id_file
    self.inference = inference or self.model.inference
    self.desc=desc

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
    evaluate_args["desc"] = self.desc
    # Evaluate
    eval_scores = []
    for eval_metric in self.eval_metrics:
      evaluate_args["evaluator"] = eval_metric
      eval_scores.append(xnmt.xnmt_evaluate.xnmt_evaluate(**evaluate_args))
    # Calculate the reference file size
    ref_words_cnt = 0
    if type(self.model.trg_reader) == TreeReader:
      with open(self.ref_file, 'r', encoding='utf-8') as myfile:
        for line in myfile:
          ref_words_cnt += len(line.split())
    else:
      for ref_sent in self.model.trg_reader.read_sents(self.ref_file):
        ref_words_cnt += self.model.trg_reader.count_words(ref_sent)
        ref_words_cnt += 0
    return eval_scores, ref_words_cnt

