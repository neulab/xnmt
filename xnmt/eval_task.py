from typing import Sequence, Union, Optional

from xnmt.settings import settings

import dynet as dy

from xnmt.evaluator import Evaluator
from xnmt.generator import GeneratorModel
from xnmt.inference import SimpleInference
import xnmt.input_reader
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.evaluator import LossScore
from xnmt.loss import LossBuilder, LossScalarBuilder
from xnmt.util import OneOrSeveral
import xnmt.xnmt_evaluate

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

  @serializable_init
  def __init__(self, src_file, ref_file, model=Ref("model"),
                batcher=Ref("train.batcher", default=None),
                loss_calculator=None, max_src_len=None, max_trg_len=None,
                desc=None):
    self.model = model
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())
    self.src_file = src_file
    self.ref_file = ref_file
    self.batcher = batcher
    self.src_data = None
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.desc=desc

  def eval(self):
    if self.src_data == None:
      self.src_data, self.ref_data, self.src_batches, self.ref_batches = \
        xnmt.input_reader.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                        self.src_file, self.ref_file, batcher=self.batcher,
                                        max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
    loss_val = LossScalarBuilder()
    ref_words_cnt = 0
    for src, trg in zip(self.src_batches, self.ref_batches):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)

      loss_builder = LossBuilder()
      standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
      additional_loss = self.model.calc_additional_loss(standard_loss)
      loss_builder.add_loss("standard_loss", standard_loss)
      loss_builder.add_loss("additional_loss", additional_loss)

      ref_words_cnt += self.model.trg_reader.count_words(trg)
      loss_val += loss_builder.get_loss_stats()

    loss_stats = {k: v/ref_words_cnt for k, v in loss_val.items()}

    try:
      return LossScore(loss_stats[self.model.get_primary_loss()], loss_stats=loss_stats, desc=self.desc), ref_words_cnt
    except KeyError:
      raise RuntimeError("Did you wrap your loss calculation with LossBuilder({'primary_loss': loss_value}) ?")

class AccuracyEvalTask(Serializable):
  '''
  A task that does evaluation of some measure of accuracy.

  Args:
    src_file: path(s) to read source file from
    ref_file: path(s) to read reference file from
    hyp_file: path to write hypothesis file to
    model: generator model to generate hypothesis with
    eval_metrics: list of evaluation metrics (list of Evaluator objects or string of comma-separated shortcuts)
    inference: inference object
    candidate_id_file (str):
    desc: human-readable description passed on to resulting score objects
  '''

  yaml_tag = '!AccuracyEvalTask'

  @serializable_init
  def __init__(self, src_file: OneOrSeveral[str], ref_file: OneOrSeveral[str], hyp_file: str,
               model: GeneratorModel = Ref("model"), eval_metrics: Union[str, Sequence[Evaluator]] = "bleu",
               inference: Optional[SimpleInference] = None, candidate_id_file: Optional[str] = None,
               desc: Optional = None):
    self.model = model
    if isinstance(eval_metrics, str):
      eval_metrics = [xnmt.xnmt_evaluate.eval_shortcuts[shortcut]() for shortcut in eval_metrics.split(",")]
    self.eval_metrics = eval_metrics
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

    # Evaluate
    eval_scores = xnmt.xnmt_evaluate.xnmt_evaluate(hyp_file=self.hyp_file, ref_file=self.ref_file, desc=self.desc,
                                                   evaluators=self.eval_metrics)

    # Calculate the reference file size
    ref_words_cnt = 0
    for ref_sent in self.model.trg_reader.read_sents(self.ref_file):
      ref_words_cnt += self.model.trg_reader.count_words(ref_sent)
      ref_words_cnt += 0
    return eval_scores, ref_words_cnt

