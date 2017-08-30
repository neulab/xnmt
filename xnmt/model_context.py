import dynet as dy
import os
from serializer import Serializable

class ModelContext(Serializable):
  yaml_tag = u'!ModelContext'
  def __init__(self):
    self.dropout = 0.0
    self.weight_noise = 0.0
    self.default_layer_dim = 512
    self.dynet_param_collection = None
    self.serialize_params = ["dropout", "weight_noise", "default_layer_dim"]
  def update(self, other):
    for param in self.serialize_params:
      setattr(self, param, getattr(other, param))

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
