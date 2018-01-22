import dynet as dy
import os
from xnmt.serialize.serializable import Serializable

class XnmtGlobal(Serializable):
  yaml_tag = u'!XnmtGlobal'
  def __init__(self, model_file=None, out_file=None, err_file=None, dropout = 0.0,
               weight_noise = 0.0, default_layer_dim = 512, save_num_checkpoints=1,
               eval_only=False, commandline_args=None,
               dynet_param_collection = None):
    self.model_file = model_file
    self.out_file = out_file
    self.err_file = err_file
    self.dropout = dropout
    self.weight_noise = weight_noise
    self.default_layer_dim = default_layer_dim
    self.model_file = None
    self.eval_only = eval_only
    self.dynet_param_collection = dynet_param_collection or PersistentParamCollection(model_file, save_num_checkpoints)
    self.commandline_args = commandline_args
  def get_model_file(self, exp_name):
    return getattr(self, "model_file", f"{exp_name}.mod")
  def get_out_file(self, exp_name):
    return getattr(self, "out_file", f"{exp_name}.out")
  def get_err_file(self, exp_name):
    return getattr(self, "err_file", f"{exp_name}.err")

class PersistentParamCollection(object):
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
    print("WARNING: reverting a non-persistent param collection has no effect")
  def save(self, fname=None):
    print("WARNING: saving a non-persistent param collection has no effect")
  def remove_existing_history(self):
    print("WARNING: editing history of a non-persistent param collection has no effect")
  def shift_safed_checkpoints(self):
    print("WARNING: editing history of a non-persistent param collection has no effect")
  def load_from_data_file(self, datafile):
    self.param_col.populate(datafile)
