import logging
logger = logging.getLogger('xnmt')

from xnmt.exp_global import ExpGlobal
from xnmt.serialize.serializable import Serializable, bare

class Experiment(Serializable):
  '''
  A default experiment that performs preprocessing, training, and evaluation.
  '''

  yaml_tag = u'!Experiment'

  def __init__(self, exp_global=bare(ExpGlobal), load=None, overwrite=None, preproc=None,
               model=None, train=None, evaluate=None, random_search_report=None):
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
        