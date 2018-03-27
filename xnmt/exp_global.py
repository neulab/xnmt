import logging
logger = logging.getLogger('xnmt')

from simple_settings import settings

from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.serializer import serializable_init
from xnmt.param_collection import ParamManager
from xnmt.param_init import ZeroInitializer, GlorotInitializer

class ExpGlobal(Serializable):
  """
  An object that holds global settings that can be referenced by components wherever appropriate.
  Also sets up the global DyNet parameter collection.
  
  Args:
    model_file (str): Location to write model file to
    log_file (str): Location to write log file to
    dropout (float): Default dropout probability that should be used by supporting components but can be overwritten
    weight_noise (float): Default weight noise level that should be used by supporting components but can be overwritten
    default_layer_dim (int): Default layer dimension that should be used by supporting components but can be overwritten
    param_init (ParamInitializer): Default parameter initializer that should be used by supporting components but can be overwritten
    bias_init (ParamInitializer): Default initializer for bias parameters that should be used by supporting components but can be overwritten
    save_num_checkpoints (int): save DyNet parameters for the most recent n checkpoints, useful for model averaging/ensembling
    eval_only (bool): If True, skip the training loop
    commandline_args (Namespace): Holds commandline arguments with which XNMT was launched
  """
  yaml_tag = '!ExpGlobal'

  @serializable_init
  def __init__(self,
               model_file=settings.DEFAULT_MOD_PATH,
               log_file=settings.DEFAULT_LOG_PATH,
               dropout = 0.3,
               weight_noise = 0.0,
               default_layer_dim = 512,
               param_init=bare(GlorotInitializer),
               bias_init=bare(ZeroInitializer),
               save_num_checkpoints=1,
               eval_only=False,
               commandline_args=None):
    # TODO: want to resolve all of these via references rather than passing the exp_global object itself.
    # once that's done, can remove the below attribute assignments
    self.model_file = model_file
    self.log_file = log_file
    self.dropout = dropout
    self.weight_noise = weight_noise
    self.default_layer_dim = default_layer_dim
    self.param_init = param_init
    self.bias_init = bias_init
    self.eval_only = eval_only
    self.commandline_args = commandline_args
    self.save_num_checkpoints = save_num_checkpoints
