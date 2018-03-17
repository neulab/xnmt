import logging
logger = logging.getLogger('xnmt')

from xnmt.exp_global import ExpGlobal
from xnmt.serialize.serializable import Serializable, bare, serializable_init

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.
  
  Args:
    exp_global (ExpGlobal): global experiment settings
    load (str): path to load a serialized experiment from (if given, only overwrite but no other arguments can be specified)
    overwrite (list): to be combined with ``load``. list of dictionaries for overwriting individual parts, with dictionaries looking like e.g. ``{"path": exp_global.eval_only, "val": True}``
    preproc (PreprocRunner): carry out preprocessing if specified
    model (GeneratorModel): The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train (TrainingRegimen): The training regimen defines the training loop.
    evaluate (List[EvalTask]): list of tasks to evaluate the model after training finishes.
    random_search_report (dict): When random search is used, this holds the settings that were randomly drawn for documentary purposes.
  '''

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self, exp_global=bare(ExpGlobal), load=None, overwrite=None, preproc=None,
               model=None, train=None, evaluate=None, random_search_report=None):
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

