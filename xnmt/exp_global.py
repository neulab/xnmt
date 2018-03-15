import logging
logger = logging.getLogger('xnmt')
import os

from simple_settings import settings
import dynet as dy

from xnmt.serialize.serializable import Serializable, bare
from xnmt.param_init import ZeroInitializer, GlorotInitializer, ParamInitializer

class ExpGlobal(Serializable):
  """
  An object that holds global settings that can be used by components wherever appropriate. Also holds the DyNet parameter collection.
  
  Args:
    model_file: Location to write model file to
    log_file: Location to write log file to
    dropout: Default dropout probability that should be used by supporting components but can be overwritten
    weight_noise: Default weight noise level that should be used by supporting components but can be overwritten
    default_layer_dim: Default layer dimension that should be used by supporting components but can be overwritten
    param_init: Default parameter initializer that should be used by supporting components but can be overwritten
    bias_init: Default initializer for bias parameters that should be used by supporting components but can be overwritten
    save_num_checkpoints (int): save DyNet parameters for the most recent n checkpoints, useful for model averaging/ensembling
    eval_only (bool): If True, skip the training loop
    commandline_args: Holds commandline arguments with which XNMT was launched
    dynet_param_collection (PersistentParamCollection): Manages DyNet weights
  """
  yaml_tag = '!ExpGlobal'
  def __init__(self,
               model_file:str=settings.DEFAULT_MOD_PATH,
               log_file:str=settings.DEFAULT_LOG_PATH,
               dropout:float = 0.3,
               weight_noise:float = 0.0,
               default_layer_dim:int = 512,
               param_init:ParamInitializer=bare(GlorotInitializer),
               bias_init:ParamInitializer=bare(ZeroInitializer),
               save_num_checkpoints:int=1,
               eval_only:bool=False,
               commandline_args=None,
               dynet_param_collection = None):
    self.model_file = model_file
    self.log_file = log_file
    self.dropout = dropout
    self.weight_noise = weight_noise
    self.default_layer_dim = default_layer_dim
    self.param_init = param_init
    self.bias_init = bias_init
    self.model_file = None
    self.eval_only = eval_only
    self.dynet_param_collection = dynet_param_collection or PersistentParamCollection(model_file, save_num_checkpoints)
    self.commandline_args = commandline_args

class PersistentParamCollection(object):
  """
  A persistent DyNet parameter collection.

  Args:
    model_file (str): file name of the model. Parameters will be written to this filename with ".data" appended
    save_num_checkpoint (int): keep the most recent this many checkpoints, by writing ".data.1" files etc.
  """
  def __init__(self, model_file, save_num_checkpoints=1):
    self.model_file = model_file
    self.param_col = dy.Model()
    self.is_saved = False
    assert save_num_checkpoints >= 1 or (model_file is None and save_num_checkpoints==0)
    if save_num_checkpoints>0: self.data_files = [self.model_file + '.data']
    for i in range(1,save_num_checkpoints):
      self.data_files.append(self.model_file + '.data.' + str(i))
  def revert_to_best_model(self):
    self.param_col.populate(self.model_file + '.data')
  def save(self, fname=None):
    if fname: assert fname == self.data_files[0], "%s != %s" % (fname + '.data', self.data_files[0])
    if not self.is_saved:
      self.remove_existing_history()
    self.shift_safed_checkpoints()
    self.param_col.save(self.data_files[0])
    self.is_saved = True
  def remove_existing_history(self):
    for fname in self.data_files[1:]:
      if os.path.exists(fname):
        os.remove(fname)
  def shift_safed_checkpoints(self):
    for i in range(len(self.data_files)-1)[::-1]:
      if os.path.exists(self.data_files[i]):
        os.rename(self.data_files[i], self.data_files[i+1])
  def load_from_data_file(self, datafile):
    self.param_col.populate(datafile)

class NonPersistentParamCollection(object):
  def __init__(self):
    self.param_col = dy.Model()
    self.model_file = None
  def revert_to_best_model(self):
    logger.warning("reverting a non-persistent param collection has no effect")
  def save(self, fname=None):
    logger.warning("saving a non-persistent param collection has no effect")
  def remove_existing_history(self):
    logger.warning("editing history of a non-persistent param collection has no effect")
  def shift_safed_checkpoints(self):
    logger.warning("editing history of a non-persistent param collection has no effect")
  def load_from_data_file(self, datafile):
    self.param_col.populate(datafile)
