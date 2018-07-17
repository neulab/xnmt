from typing import Sequence, Union, Optional, Any

from xnmt.settings import settings

import dynet as dy

from xnmt.batcher import Batcher
from xnmt.evaluator import Evaluator
from xnmt import model_base
import xnmt.inference
import xnmt.input_reader
from xnmt.persistence import serializable_init, Serializable, Ref, bare
from xnmt.loss_calculator import LossCalculator, AutoRegressiveMLELoss
from xnmt.evaluator import LossScore
from xnmt.loss import FactoredLossExpr, FactoredLossVal
import xnmt.xnmt_evaluate
from xnmt import util

class EvalTask(object):
  """
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  """
  def eval(self):
    raise NotImplementedError("EvalTask.eval() needs to be implemented in child classes")

class LossEvalTask(EvalTask, Serializable):
  """
  A task that does evaluation of the loss function.

  Args:
    src_file: source file name
    ref_file: reference file name
    model: generator model to use for inference
    batcher: batcher to use
    loss_calculator: loss calculator
    max_src_len: omit sentences with source length greater than specified number
    max_trg_len: omit sentences with target length greater than specified number
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    desc: description to pass on to computed score objects
  """
  yaml_tag = '!LossEvalTask'

  @serializable_init
  def __init__(self, src_file: str, ref_file: Optional[str] = None, model: 'model_base.GeneratorModel' = Ref("model"),
               batcher: Batcher = Ref("train.batcher", default=bare(xnmt.batcher.SrcBatcher, batch_size=32)),
               loss_calculator: LossCalculator = bare(AutoRegressiveMLELoss), max_src_len: Optional[int] = None,
               max_trg_len: Optional[int] = None,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"), desc: Any = None):
    self.model = model
    self.loss_calculator = loss_calculator
    self.src_file = src_file
    self.ref_file = ref_file
    self.batcher = batcher
    self.src_data = None
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.loss_comb_method = loss_comb_method
    self.desc=desc

  def eval(self) -> 'EvalScore':
    """
    Perform evaluation task.

    Returns:
      Evaluated score
    """
    self.model.set_train(False)
    if self.src_data is None:
      self.src_data, self.ref_data, self.src_batches, self.ref_batches = \
        xnmt.input_reader.read_parallel_corpus(src_reader=self.model.src_reader,
                                               trg_reader=self.model.trg_reader,
                                               src_file=self.src_file,
                                               trg_file=self.ref_file,
                                               batcher=self.batcher,
                                               max_src_len=self.max_src_len,
                                               max_trg_len=self.max_trg_len)
    loss_val = FactoredLossVal()
    ref_words_cnt = 0
    for src, trg in zip(self.src_batches, self.ref_batches):
      with util.ReportOnException({"src": src, "trg": trg, "graph": dy.print_text_graphviz}):
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)

        loss_builder = FactoredLossExpr()
        standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
        additional_loss = self.model.calc_additional_loss(trg, self.model, standard_loss)
        loss_builder.add_factored_loss_expr(standard_loss)
        loss_builder.add_factored_loss_expr(additional_loss)

        ref_words_cnt += sum([trg_i.len_unpadded() for trg_i in trg])
        loss_val += loss_builder.get_factored_loss_val(comb_method=self.loss_comb_method)

    loss_stats = {k: v/ref_words_cnt for k, v in loss_val.items()}

    try:
      return LossScore(loss_stats[self.model.get_primary_loss()],
                       loss_stats=loss_stats,
                       num_ref_words = ref_words_cnt,
                       desc=self.desc)
    except KeyError:
      raise RuntimeError("Did you wrap your loss calculation with FactoredLossExpr({'primary_loss': loss_value}) ?")

class AccuracyEvalTask(EvalTask, Serializable):
  """
  A task that does evaluation of some measure of accuracy.

  Args:
    src_file: path(s) to read source file(s) from
    ref_file: path(s) to read reference file(s) from
    hyp_file: path to write hypothesis file to
    model: generator model to generate hypothesis with
    eval_metrics: list of evaluation metrics (list of Evaluator objects or string of comma-separated shortcuts)
    inference: inference object
    desc: human-readable description passed on to resulting score objects
  """

  yaml_tag = '!AccuracyEvalTask'

  @serializable_init
  def __init__(self, src_file: Union[str,Sequence[str]], ref_file: Union[str,Sequence[str]], hyp_file: str,
               model: 'model_base.GeneratorModel' = Ref("model"), eval_metrics: Union[str, Sequence[Evaluator]] = "bleu",
               inference: Optional[xnmt.inference.Inference] = None, desc: Any = None):
    self.model = model
    if isinstance(eval_metrics, str):
      eval_metrics = [xnmt.xnmt_evaluate.eval_shortcuts[shortcut]() for shortcut in eval_metrics.split(",")]
    elif not isinstance(eval_metrics, str): eval_metrics = [eval_metrics]
    self.eval_metrics = eval_metrics
    self.src_file = src_file
    self.ref_file = ref_file
    self.hyp_file = hyp_file
    self.inference = inference or self.model.inference
    self.desc=desc

  def eval(self):
    self.model.set_train(False)
    self.inference.perform_inference(generator=self.model,
                                     src_file=self.src_file,
                                     trg_file=self.hyp_file,
                                     ref_file_to_report=self.ref_file)
    # Evaluate
    eval_scores = xnmt.xnmt_evaluate.xnmt_evaluate(hyp_file=self.hyp_file, ref_file=self.ref_file, desc=self.desc,
                                                   evaluators=self.eval_metrics)

    return eval_scores

class DecodingEvalTask(EvalTask, Serializable):
  """
  A task that does performs decoding without comparing against a reference.

  Args:
    src_file: path(s) to read source file(s) from
    hyp_file: path to write hypothesis file to
    model: generator model to generate hypothesis with
    inference: inference object
  """

  yaml_tag = '!DecodingEvalTask'

  @serializable_init
  def __init__(self, src_file: Union[str,Sequence[str]], hyp_file: str, model: 'model_base.GeneratorModel' = Ref("model"),
               inference: Optional[xnmt.inference.Inference] = None):

    self.model = model
    self.src_file = src_file
    self.hyp_file = hyp_file
    self.inference = inference or self.model.inference

  def eval(self):
    self.model.set_train(False)
    self.inference.perform_inference(generator=self.model,
                                     src_file=self.src_file,
                                     trg_file=self.hyp_file)
    return None
