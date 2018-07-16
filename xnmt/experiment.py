from typing import List, Optional

from xnmt import logger
from xnmt.exp_global import ExpGlobal
from xnmt.eval_task import EvalTask
from xnmt.model_base import TrainableModel
from xnmt.param_collection import ParamManager, RevertingUnsavedModelException
from xnmt.preproc_runner import PreprocRunner
from xnmt.training_regimen import TrainingRegimen
from xnmt.persistence import serializable_init, Serializable, bare

class Experiment(Serializable):
  """
  A default experiment that performs preprocessing, training, and evaluation.

  The initializer calls ParamManager.populate(), meaning that model construction should be finalized at this point.
  __call__() runs the individual steps.

  Args:
    exp_global: global experiment settings
    preproc: carry out preprocessing if specified
    model: The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train: The training regimen defines the training loop.
    evaluate: list of tasks to evaluate the model after training finishes.
    random_search_report: When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  """

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self,
               exp_global:Optional[ExpGlobal] = bare(ExpGlobal),
               preproc:Optional[PreprocRunner] = None,
               model:Optional[TrainableModel] = None,
               train:Optional[TrainingRegimen] = None,
               evaluate:Optional[List[EvalTask]] = None,
               random_search_report:Optional[dict] = None) -> None:
    self.exp_global = exp_global
    self.preproc = preproc
    self.model = model
    self.train = train
    self.evaluate = evaluate

    if random_search_report:
      logger.info(f"> instantiated random parameter search: {random_search_report}")

  def __call__(self, save_fct):
    """
    Launch training loop, followed by final evaluation.
    """
    eval_scores = ["Not evaluated"]
    if self.train:
      logger.info("> Training")
      self.train.run_training(save_fct = save_fct)
      logger.info('reverting learned weights to best checkpoint..')
      try:
        ParamManager.param_col.revert_to_best_model()
      except RevertingUnsavedModelException:
        pass

    evaluate_args = self.evaluate
    if evaluate_args:
      logger.info("> Performing final evaluation")
      eval_scores = []
      for evaluator in evaluate_args:
        eval_score = evaluator.eval()
        if type(eval_score) == list:
          eval_scores.extend(eval_score)
        else:
          eval_scores.append(eval_score)

    return eval_scores

