from typing import Any, Callable, Dict, List, Optional, Sequence
import numbers

from xnmt.settings import settings

from xnmt import logger, param_collections, param_initializers, preproc
from xnmt.eval import metrics, tasks as eval_tasks
from xnmt.models import base as models_base
from xnmt.train import regimens
from xnmt.persistence import serializable_init, Serializable, bare

class ExpGlobal(Serializable):
  """
  An object that holds global settings that can be referenced by components wherever appropriate.

  Args:
    model_file: Location to write model file to
    log_file: Location to write log file to
    dropout: Default dropout probability that should be used by supporting components but can be overwritten
    weight_noise: Default weight noise level that should be used by supporting components but can be overwritten
    default_layer_dim: Default layer dimension that should be used by supporting components but can be overwritten
    param_init: Default parameter initializer that should be used by supporting components but can be overwritten
    bias_init: Default initializer for bias parameters that should be used by supporting components but can be overwritten
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
    save_num_checkpoints: save DyNet parameters for the most recent n checkpoints, useful for model averaging/ensembling
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    commandline_args: Holds commandline arguments with which XNMT was launched
    placeholders: these will be used as arguments for a format() call applied to every string in the config.
                  For example, ``placeholders: {"PATH":"/some/path"} will cause each occurence of ``"{PATH}"`` in a
                  string to be replaced by ``"/some/path"``. As a special variable, ``EXP_DIR`` can be specified to
                  overwrite the default location for writing models, logs, and other files.
  """
  yaml_tag = '!ExpGlobal'

  @serializable_init
  def __init__(self,
               model_file: str = settings.DEFAULT_MOD_PATH,
               log_file: str = settings.DEFAULT_LOG_PATH,
               dropout: numbers.Real = 0.3,
               weight_noise: numbers.Real = 0.0,
               default_layer_dim: numbers.Integral = 512,
               param_init: param_initializers.ParamInitializer = bare(param_initializers.GlorotInitializer),
               bias_init: param_initializers.ParamInitializer = bare(param_initializers.ZeroInitializer),
               truncate_dec_batches: bool = False,
               save_num_checkpoints: numbers.Integral = 1,
               loss_comb_method: str = "sum",
               commandline_args: dict = {},
               placeholders: Dict[str, Any] = {}) -> None:
    self.model_file = model_file
    self.log_file = log_file
    self.dropout = dropout
    self.weight_noise = weight_noise
    self.default_layer_dim = default_layer_dim
    self.param_init = param_init
    self.bias_init = bias_init
    self.truncate_dec_batches = truncate_dec_batches
    self.commandline_args = commandline_args
    self.save_num_checkpoints = save_num_checkpoints
    self.loss_comb_method = loss_comb_method
    self.placeholders = placeholders

class Experiment(Serializable):
  """
  A default experiment that performs preprocessing, training, and evaluation.

  The initializer calls ParamManager.populate(), meaning that model construction should be finalized at this point.
  __call__() runs the individual steps.

  Args:
    name: name of experiment
    exp_global: global experiment settings
    preproc: carry out preprocessing if specified
    model: The main model. In the case of multitask training, several models must be specified, in which case the models will live not here but inside the training task objects.
    train: The training regimen defines the training loop.
    evaluate: list of tasks to evaluate the model after training finishes.
    random_search_report: When random search is used, this holds the settings that were randomly drawn for documentary purposes.
    status: Status of the experiment, will be automatically set to "done" in saved model if the experiment has finished running.
  """

  yaml_tag = '!Experiment'

  @serializable_init
  def __init__(self,
               name: str,
               exp_global: Optional[ExpGlobal] = bare(ExpGlobal),
               preproc: Optional[preproc.PreprocRunner] = None,
               model: Optional[models_base.TrainableModel] = None,
               train: Optional[regimens.TrainingRegimen] = None,
               evaluate: Optional[List[eval_tasks.EvalTask]] = None,
               random_search_report: Optional[dict] = None,
               status: Optional[str] = None) -> None:
    self.name = name
    self.exp_global = exp_global
    self.preproc = preproc
    self.model = model
    self.train = train
    self.evaluate = evaluate
    self.status = status

    if random_search_report:
      logger.info(f"> instantiated random parameter search: {random_search_report}")

  def __call__(self, save_fct: Callable) -> Sequence[metrics.EvalScore]:
    """
    Launch training loop, followed by final evaluation.
    """
    eval_scores = ["Not evaluated"]
    if self.status != "done":
      if self.train:
        logger.info("> Training")
        self.train.run_training(save_fct = save_fct)
        logger.info('reverting learned weights to best checkpoint..')
        try:
          param_collections.ParamManager.param_col.revert_to_best_model()
        except param_collections.RevertingUnsavedModelException:
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

      self.save_processed_arg("status", "done")
      save_fct()
    else:
      logger.info("Experiment already finished, skipping.")

    return eval_scores

