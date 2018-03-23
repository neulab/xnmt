import logging
logger = logging.getLogger('xnmt')
from typing import List, Optional

from xnmt.exp_global import ExpGlobal
from xnmt.preproc_runner import PreprocRunner
from xnmt.serialize.serializable import Serializable, bare
from xnmt.training_regimen import TrainingRegimen
from xnmt.generator import GeneratorModel
from xnmt.eval_task import EvalTask

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.
  
  Args:
    exp_global: global experiment settings
    load: to be combined with ``overwrite``. Path to load a serialized experiment from (if given, only overwrite but no other arguments can be specified)
    overwrite: to be combined with ``load``. List of dictionaries for overwriting individual parts, with dictionaries looking like e.g. ``{"path": exp_global.eval_only, "val": True}``
    preproc: carry out preprocessing if specified
    model: The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train: The training regimen defines the training loop.
    evaluate: list of tasks to evaluate the model after training finishes.
    random_search_report: When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  '''

  yaml_tag = '!Experiment'

  def __init__(self,
               exp_global = bare(ExpGlobal),
               load:Optional[str] = None,
               overwrite:Optional[str] = None,
               preproc:PreprocRunner = None,
               model:Optional[GeneratorModel] = None,
               train:TrainingRegimen = None,
               evaluate:Optional[List[EvalTask]] = None,
               random_search_report:Optional[dict] = None) -> None:
    """
    This is called after all other components have been initialized, so we can safely load DyNet weights here. 
    """
    self.exp_global = exp_global
    self.load = load
    self.overwrite = overwrite
    self.preproc = preproc
    self.model = model
    self.train = train
    self.evaluate = evaluate
    if load:
      exp_global.dynet_param_collection.load_from_data_file(f"{load}.data")
      logger.info(f"> populated DyNet weights from {load}.data")

    if random_search_report:
      logger.info(f"> instantiated random parameter search: {random_search_report}")

  def __call__(self, save_fct):
    """
    Launch training loop, followed by final evaluation.
    """
    eval_scores = "Not evaluated"
    eval_only = self.exp_global.eval_only
    if not eval_only:
      logger.info("> Training")
      self.train.run_training(save_fct = save_fct)
      logger.info('reverting learned weights to best checkpoint..')
      self.exp_global.dynet_param_collection.revert_to_best_model()

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

