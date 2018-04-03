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
  initialized = False

  @staticmethod
  def init_param_col():
    ParamManager.param_col = ParamCollection()
    ParamManager.load_paths = []
    ParamManager.initialized = True

  @staticmethod
  def set_save_file(file_name, save_num_checkpoints=1):
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    ParamManager.param_col.model_file = file_name
    ParamManager.param_col.save_num_checkpoints = save_num_checkpoints

  @staticmethod
  def add_load_path(data_file):
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    ParamManager.load_paths.append(data_file)

  @staticmethod
  def populate():
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    populated_subcols = []
    for subcol_name in ParamManager.param_col.subcols:
      for load_path in ParamManager.load_paths:
        data_file = os.path.join(load_path, subcol_name)
        if os.path.isfile(data_file):
          ParamManager.param_col.load_subcol_from_data_file(subcol_name, data_file)
          populated_subcols.append(subcol_name)
    if len(ParamManager.param_col.subcols) == len(populated_subcols):
      logger.info(f"> populated DyNet weights of all components from given data files")
    elif len(populated_subcols)==0:
      logger.info(f"> use randomly initialized DyNet weights of all components")
    else:
      logger.info(f"> populated a subset of DyNet weights from given data files: {populated_subcols}.\n"
                  f"  Did not populate {ParamManager.param_col.subcols.keys() - set(populated_subcols)}")

  @staticmethod
  def my_params(subcol_owner):
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    """Creates a dedicated parameter subcollection for a serializable object. This should only be called from the
    __init__ method of a Serializable.

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
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
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
    assert data_dir.endswith(".data") or data_dir.split(".")[-2] == "data"
    try:
      dir_contents = os.listdir(data_dir)
      for old_file in dir_contents:
        spl = old_file.split(".")
        # make sure we're only deleting files with the expected filenames
        if len(spl)==2:
          if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", spl[0]):
            if re.match(r"^[0-9a-f]{8}$", spl[1]):
              os.remove(os.path.join(data_dir, old_file))
    except NotADirectoryError:
      os.remove(data_dir)
  def _shift_saved_checkpoints(self):
    if os.path.exists(self._data_files[-1]):
      self._remove_data_dir(self._data_files[-1])
    for i in range(len(self._data_files)-1)[::-1]:
      if os.path.exists(self._data_files[i]):
        os.rename(self._data_files[i], self._data_files[i+1])
