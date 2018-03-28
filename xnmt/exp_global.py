import logging
logger = logging.getLogger('xnmt')
from typing import Dict

from simple_settings import settings

from xnmt.serialize.serializable import Serializable, bare
from xnmt.serialize.serializer import serializable_init
from xnmt.param_init import ZeroInitializer, GlorotInitializer, ParamInitializer


class ExpGlobal(Serializable):
  """
  An object that holds global settings that can be referenced by components wherever appropriate.
  Also sets up the global DyNet parameter collection.
  
  Args:
    model_file: Location to write model file to
    log_file: Location to write log file to
    dropout: Default dropout probability that should be used by supporting components but can be overwritten
    weight_noise: Default weight noise level that should be used by supporting components but can be overwritten
    default_layer_dim: Default layer dimension that should be used by supporting components but can be overwritten
    param_init: Default parameter initializer that should be used by supporting components but can be overwritten
    bias_init: Default initializer for bias parameters that should be used by supporting components but can be overwritten
    save_num_checkpoints: save DyNet parameters for the most recent n checkpoints, useful for model averaging/ensembling
    commandline_args: Holds commandline arguments with which XNMT was launched
    placeholders: these will be used as arguments for a format() call applied to every string in the config.
                                  For example, ``placeholders: {"PATH":"/some/path"} will cause each occurence of ``"{PATH}"`` in a string
                                  to be replaced by ``"/some/path"``.
  """
  yaml_tag = '!ExpGlobal'

  @serializable_init
  def __init__(self,
               model_file: str = settings.DEFAULT_MOD_PATH,
               log_file: str = settings.DEFAULT_LOG_PATH,
               dropout: float = 0.3,
               weight_noise: float = 0.0,
               default_layer_dim: int = 512,
               param_init: ParamInitializer = bare(GlorotInitializer),
               bias_init: ParamInitializer = bare(ZeroInitializer),
               save_num_checkpoints: int = 1,
               commandline_args=None,
               placeholders: Dict[str, str] = {}):

    # TODO: want to resolve all of these via references rather than passing the exp_global object itself.
    # once that's done, can remove the below attribute assignments

    self.model_file = model_file
    self.log_file = log_file
    self.dropout = dropout
    self.weight_noise = weight_noise
    self.default_layer_dim = default_layer_dim
    self.param_init = param_init
    self.bias_init = bias_init
    self.commandline_args = commandline_args
    self.save_num_checkpoints = save_num_checkpoints
    self.placeholders = placeholders
