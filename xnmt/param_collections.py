import os
import re

import dynet as dy

from xnmt import logger

class ParamManager(object):
  """
  A static class that manages the currently loaded DyNet parameters of all components.

  Responsibilities are registering of all components that use DyNet parameters and loading pretrained parameters.
  Components can register parameters by calling ParamManager.my_params(self) from within their __init__() method.
  This allocates a subcollection with a unique identifier for this component. When loading previously saved parameters,
  one or several paths are specified to look for the corresponding saved DyNet collection named after this identifier.
  """
  initialized = False

  @staticmethod
  def init_param_col() -> None:
    """
    Initializes or resets the parameter collection.

    This must be invoked before every time a new model is loaded (e.g. on startup and between consecutive experiments).
    """
    ParamManager.param_col = ParamCollection()
    ParamManager.load_paths = []
    ParamManager.initialized = True

  @staticmethod
  def add_load_path(data_file: str) -> None:
    """
    Add new data directory path to load from.

    When calling populate(), pretrained parameters from all directories added in this way are searched for the
    requested component identifiers.

    Args:
      data_file: a data directory (usually named ``*.data``) containing DyNet parameter collections.
    """
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    if not data_file in ParamManager.load_paths: ParamManager.load_paths.append(data_file)

  @staticmethod
  def populate() -> None:
    """
    Populate the parameter collections.

    Searches the given data paths and loads parameter collections if they exist, otherwise leave parameters in their
    randomly initialized state.
    """
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
                  f"  Did not populate {ParamManager.param_col.subcols.keys() - set(populated_subcols)}.\n"
                  f"  If partial population was not intended, likely the unpopulated component or its owner"
                  f"   does not adhere to the Serializable protocol correctly, see documentation:\n"
                  f"   http://xnmt.readthedocs.io/en/latest/writing_xnmt_classes.html#using-serializable-subcomponents")
    logger.info(f"  DyNet param count: {ParamManager.param_col._param_col.parameter_count()}")

  @staticmethod
  def my_params(subcol_owner) -> dy.ParameterCollection:
    """Creates a dedicated parameter subcollection for a serializable object.

    This should only be called from the __init__ method of a Serializable.

    Args:
      subcol_owner (Serializable): The object which is requesting to be assigned a subcollection.

    Returns:
      The assigned subcollection.
    """
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    assert not getattr(subcol_owner, "init_completed", False), \
      f"my_params(obj) cannot be called after obj.__init__() has completed. Conflicting obj: {subcol_owner}"
    if not hasattr(subcol_owner, "xnmt_subcol_name"):
      raise ValueError(f"{subcol_owner} does not have an attribute 'xnmt_subcol_name'.\n"
                       f"Did you forget to wrap the __init__() in @serializable_init ?")
    subcol_name = subcol_owner.xnmt_subcol_name
    subcol = ParamManager.param_col.add_subcollection(subcol_owner, subcol_name)
    subcol_owner.save_processed_arg("xnmt_subcol_name", subcol_name)
    return subcol

  @staticmethod
  def global_collection() -> dy.ParameterCollection:
    """ Access the top-level parameter collection, including all parameters.

    Returns:
      top-level DyNet parameter collection
    """
    assert ParamManager.initialized, "must call ParamManager.init_param_col() first"
    return ParamManager.param_col._param_col

class ParamCollection(object):

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._save_num_checkpoints = 1
    self._model_file = None
    self._param_col = dy.Model()
    self._is_saved = False
    self.subcols = {}
    self.all_subcol_owners = set()

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

  def add_subcollection(self, subcol_owner: 'Serializable', subcol_name: str) -> dy.ParameterCollection:
    assert subcol_owner not in self.all_subcol_owners
    self.all_subcol_owners.add(subcol_owner)
    if subcol_name in self.subcols:
      raise RuntimeError(f'Duplicate subcol_name {subcol_name} found when loading')
    new_subcol = self._param_col.add_subcollection(subcol_name)
    self.subcols[subcol_name] = new_subcol
    return new_subcol

  def load_subcol_from_data_file(self, subcol_name: str, data_file: str) -> None:
    self.subcols[subcol_name].populate(data_file)

  def save(self) -> None:
    if not self._is_saved:
      self._remove_existing_history()
    self._shift_saved_checkpoints()
    if not os.path.exists(self._data_files[0]):
      os.makedirs(self._data_files[0])
    for subcol_name, subcol in self.subcols.items():
      subcol.save(os.path.join(self._data_files[0], subcol_name))
    self._is_saved = True

  def revert_to_best_model(self) -> None:
    if not self._is_saved:
      raise RevertingUnsavedModelException("revert_to_best_model() is illegal because this model has never been saved.")
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

class RevertingUnsavedModelException(Exception): pass