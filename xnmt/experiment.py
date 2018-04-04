import logging
logger = logging.getLogger('xnmt')
from typing import List, Optional

from xnmt.exp_global import ExpGlobal
from xnmt.param_collection import ParamManager
from xnmt.serialize.serializer import serializable_init
from xnmt.preproc_runner import PreprocRunner
from xnmt.serialize.serializable import Serializable, bare
from xnmt.training_regimen import TrainingRegimen
from xnmt.generator import GeneratorModel
from xnmt.eval_task import EvalTask

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.

  The initializer calls ParamManager.populate(), meaning that model construction should be finalized at this point.
  __call__() runs the individual steps.
  
  Args:
    exp_global: global experiment settings
    preproc: carry out preprocessing if specified
    model: The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    eval_zero: list of tasks to evaluate before the training starts (useful e.g. with pretrained models).
    train: The training regimen defines the training loop.
    evaluate: list of tasks to evaluate the model after training finishes.
    random_search_report: When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  '''

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self,
               exp_global:Optional[ExpGlobal] = bare(ExpGlobal),
               preproc:Optional[PreprocRunner] = None,
               model:Optional[GeneratorModel] = None,
               eval_zero:Optional[List[EvalTask]] = None,
               train:Optional[TrainingRegimen] = None,
               evaluate:Optional[List[EvalTask]] = None,
               random_search_report:Optional[dict] = None) -> None:
    self.exp_global = exp_global
    self.preproc = preproc
    self.model = model
    self.eval_zero = eval_zero
    self.train = train
    self.evaluate = evaluate

    if random_search_report:
      logger.info(f"> instantiated random parameter search: {random_search_report}")

  def __call__(self, save_fct):
    """
    Launch eval_zero -> training loop -> final evaluation (all optional).
    """
    eval_zero_scores = self.run_evaluators(self.eval_zero, "> Performing zero evaluation")
    if eval_zero_scores:
      for i in range(len(eval_zero_scores)):
        print(f"  [Zero score] {str(eval_zero_scores[i])}")

    if self.train:
      logger.info("> Training")
      self.train.run_training(save_fct = save_fct)
      logger.info('reverting learned weights to best checkpoint..')
      ParamManager.param_col.revert_to_best_model()

    eval_scores = self.run_evaluators(self.evaluate, "> Performing final evaluation") or "Not evaluated"
    return eval_scores

  def run_evaluators(self, evaluators, log_str):
    eval_scores = None
    if evaluators:
      logger.info(log_str)
      eval_scores = []
      for evaluator in evaluators:
        eval_score, _ = evaluator.eval()
        if type(eval_score) == list:
          eval_scores.extend(eval_score)
        else:
          eval_scores.append(eval_score)
    return eval_scores

