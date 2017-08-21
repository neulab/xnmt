import dynet as dy
import os

dynet_param_collection = None

model_globals = {
  "dropout" : 0.0,
  "weight_noise" : 0.0,
  "default_layer_dim" : 512,
}

def get(key):
  return model_globals.get(key)

def default_if_none(value):
  if not value:
    return get("default_layer_dim")
  else:
    return value

class PersistentParamCollection(object):
  def __init__(self, model_file, save_num_checkpoints=1):
    self.model_file = model_file
    self.param_col = dy.Model()
    self.is_saved = False
    assert save_num_checkpoints >= 1
    self.data_files = [self.model_file + '.data']
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
