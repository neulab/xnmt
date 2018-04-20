from typing import List, Optional

from xnmt import logger
from xnmt.persistence import serializable_init, Serializable, bare
import xnmt.exp_global as exp_global
import xnmt.eval_task as eval_task
import xnmt.generator as generator
import xnmt.param_collection as pc
import xnmt.preproc_runner as preproc_runner
import xnmt.training_regimen as training_regimen

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.

  The initializer calls param_collection.ParamManager.populate(), meaning that model construction should be finalized at this point.
  __call__() runs the individual steps.
  
  Args:
    exp_global: global experiment settings
    preproc: carry out preprocessing if specified
    model: The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train: The training regimen defines the training loop.
    evaluate: list of tasks to evaluate the model after training finishes.
    random_search_report: When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  '''

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self,
               exp_global:Optional[exp_global.ExpGlobal] = bare(exp_global.ExpGlobal),
               preproc:Optional[preproc_runner.PreprocRunner] = None,
               model:Optional[generator.GeneratorModel] = None,
               train:Optional[training_regimen.TrainingRegimen] = None,
               evaluate:Optional[List[eval_task.EvalTask]] = None,
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
      save_fct() # save initial model
      self.train.run_training(save_fct = save_fct)
      logger.info('reverting learned weights to best checkpoint..')
      pc.ParamManager.param_col.revert_to_best_model()

    evaluate_args = self.evaluate
    if evaluate_args:
      logger.info("> Performing final evaluation")
      eval_scores = []
      for evaluator in evaluate_args:
        eval_score, _ = evaluator.eval()
        if type(eval_score) == list:
          eval_scores.extend(eval_score)
        else:
          eval_scores.append(eval_score)

    return eval_scores

