import logging
logger = logging.getLogger('xnmt')
import os
import re

import dynet as dy

class ParamManager(object):
  """
  A static class that manages the currently loaded DyNet parameters.
  There is only one parameter manager, but it can manage parameters from multiple models via named subcollections.
  """

  @staticmethod
  def init_param_col():
    ParamManager.param_col = ParamCollection()
    ParamManager.load_paths = []

  @staticmethod
  def set_save_file(file_name, save_num_checkpoints=1):
    ParamManager.param_col.model_file = file_name
    ParamManager.param_col.save_num_checkpoints = save_num_checkpoints

  @staticmethod
  def add_load_path(data_file):
    ParamManager.load_paths.append(data_file)

  @staticmethod
  def populate():
    for subcol_name in ParamManager.param_col.subcols:
      for load_path in ParamManager.load_paths:
        data_file = os.path.join(load_path, subcol_name)
        if os.path.isfile(data_file):
          ParamManager.param_col.load_subcol_from_data_file(subcol_name, data_file)
          logger.info(f"> populated DyNet weights of {subcol_name} from {data_file}")

  @staticmethod
  def my_subcollection(subcol_owner):
    """Creates a dedicated subcollection for a serializable object. This should only be called from the __init__ method
    of a Serializable.

    Args:
      subcol_owner (Serializable): The object which is requesting to be assigned a subcollection.

    Returns:
      dynet.ParamManager: The assigned subcollection.
    """
    if not hasattr(subcol_owner, "xnmt_subcol_name"):
      raise ValueError(f"{node} does not have an attribute 'xnmt_subcol_name'.\n"
                       f"Did you forget to wrap the __init__() in @serializable_init ?")
    subcol_name = subcol_owner.xnmt_subcol_name
    subcol = ParamManager.param_col.add_subcollection(subcol_name)
    subcol_owner.overwrite_serialize_param("xnmt_subcol_name", subcol_name)
    return subcol

  @staticmethod
  def global_collection():
    """ Access the top-level parameter collection

    Returns:
      dynet.ParamCollection: top-level DyNet parameter collection
    """
    return ParamManager.param_col._param_col

class ParamCollection(object):

  def __init__(self):
    self.reset()
  def reset(self):
    self._save_num_checkpoints = 1
    self._model_file = None
    self._param_col = dy.Model()
    self._is_saved = False
    self.subcols = {}

  @property
  def save_num_checkpoints(self):
    return self._save_num_checkpoints
  @save_num_checkpoints.setter
  def save_num_checkpoints(self, value):
    self._save_num_checkpoints = value
    self._update_data_files()
  @property
  def model_file(self):
    return self._model_file
  @model_file.setter
  def model_file(self, value):
    self._model_file = value
    self._update_data_files()
  def _update_data_files(self):
    if self._save_num_checkpoints>0 and self._model_file:
      self._data_files = [self.model_file + '.data']
      for i in range(1,self._save_num_checkpoints):
        self._data_files.append(self.model_file + '.data.' + str(i))
    else:
      self._data_files = []

  def add_subcollection(self, subcol_name):
    assert subcol_name not in self.subcols
    new_subcol = self._param_col.add_subcollection(subcol_name)
    self.subcols[subcol_name] = new_subcol
    return new_subcol

  def load_subcol_from_data_file(self, subcol_name, data_file):
    self.subcols[subcol_name].populate(data_file)

  def save(self):
    if not self._is_saved:
      self._remove_existing_history()
    self._shift_saved_checkpoints()
    if not os.path.exists(self._data_files[0]):
      os.makedirs(self._data_files[0])
    for subcol_name, subcol in self.subcols.items():
      subcol.save(os.path.join(self._data_files[0], subcol_name))
    self._is_saved = True

  def revert_to_best_model(self):
    for subcol_name, subcol in self.subcols.items():
      subcol.populate(os.path.join(self._data_files[0], subcol_name))

  def _remove_existing_history(self):
    for fname in self._data_files:
      if os.path.exists(fname):
        self._remove_data_dir(fname)
  def _remove_data_dir(self, data_dir):
    dir_contents = os.listdir(data_dir)
    for old_file in dir_contents:
      spl = old_file.split(".")
      # make sure we're only deleting files with the expected filenames
      if len(spl)==2:
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", spl[0]):
          if re.match(r"^[0-9a-f]{8}$", spl[1]):
            os.remove(os.path.join(data_dir, old_file))
  def _shift_saved_checkpoints(self):
    if os.path.exists(self._data_files[-1]):
      self._remove_data_dir(self._data_files[-1])
    for i in range(len(self._data_files)-1)[::-1]:
      if os.path.exists(self._data_files[i]):
        os.rename(self._data_files[i], self._data_files[i+1])

# class PersistentParamCollection(ParamCollection):
#   """
#   A persistent DyNet parameter collection.
#
#   Args:
#     model_file (str): file name of the model. Parameters will be written to this filename with ".data" appended
#     save_num_checkpoint (int): keep the most recent this many checkpoints, by writing ".data.1" files etc.
#   """
#   def __init__(self, model_file, save_num_checkpoints=1):
#     self.model_file = model_file
#     self.param_col = dy.Model()
#     self.is_saved = False
#     assert save_num_checkpoints >= 1 or (model_file is None and save_num_checkpoints==0)
#     if save_num_checkpoints>0: self.data_files = [self.model_file + '.data']
#     for i in range(1,save_num_checkpoints):
#       self.data_files.append(self.model_file + '.data.' + str(i))
#   def revert_to_best_model(self):
#     self.param_col.populate(self.model_file + '.data')
#   def save(self, fname=None):
#     if fname: assert fname == self.data_files[0], "%s != %s" % (fname + '.data', self.data_files[0])
#     if not self.is_saved:
#       self.remove_existing_history()
#     self.shift_safed_checkpoints()
#     self.param_col.save(self.data_files[0])
#     self.is_saved = True
#   def remove_existing_history(self):
#     for fname in self.data_files[1:]:
#       if os.path.exists(fname):
#         os.remove(fname)
#   def shift_safed_checkpoints(self):
#     for i in range(len(self.data_files)-1)[::-1]:
#       if os.path.exists(self.data_files[i]):
#         os.rename(self.data_files[i], self.data_files[i+1])
#   def load_from_data_file(self, datafile):
#     self.param_col.populate(datafile)
#
# class NonPersistentParamCollection(ParamCollection):
#   def __init__(self):
#     self.param_col = dy.Model()
#     self.model_file = None
#   def revert_to_best_model(self):
#     logger.warning("reverting a non-persistent param collection has no effect")
#   def save(self, fname=None):
#     logger.warning("saving a non-persistent param collection has no effect")
#   def remove_existing_history(self):
#     logger.warning("editing history of a non-persistent param collection has no effect")
#   def shift_safed_checkpoints(self):
#     logger.warning("editing history of a non-persistent param collection has no effect")
#   def load_from_data_file(self, datafile):
#     self.param_col.populate(datafile)
