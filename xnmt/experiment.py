import logging
logger = logging.getLogger('xnmt')

from xnmt.exp_global import ExpGlobal
from xnmt.param_collection import ParamManager
from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.serializer import serializable_init

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.

  The initializer calls ParamManager.populate(), meaning that model construction should be finalized at this point.
  __call__() runs the individual steps.
  
  Args:
    exp_global (ExpGlobal): global experiment settings
    preproc (PreprocRunner): carry out preprocessing if specified
    model (GeneratorModel): The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train (TrainingRegimen): The training regimen defines the training loop.
    evaluate (List[EvalTask]): list of tasks to evaluate the model after training finishes.
    random_search_report (dict): When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  '''

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self, exp_global=bare(ExpGlobal), preproc=None,  model=None, train=None, evaluate=None,
               random_search_report=None):
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
    eval_scores = "Not evaluated"
    eval_only = self.exp_global.eval_only
    if not eval_only:
      logger.info("> Training")
      self.train.run_training(save_fct = save_fct)
      logger.info('reverting learned weights to best checkpoint..')
      ParamManager.param_col.revert_to_best_model()

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

