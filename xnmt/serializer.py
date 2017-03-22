from __future__ import print_function
import dynet as dy
import json

class JSONSerializer:
  
  '''
  Save a model to file.

  fname -- the file name
  mod -- the root model class
  param -- the model parameters
  '''
  def save_to_file(self, fname, mod, params):
    with open(fname, 'w') as f:
      json.dump(mod.to_spec(), f)
    params.save(fname + '.data')

  '''
  Load a model from a file.

  fname -- the file name
  spec -- the root model class
  param -- the model parameters
  '''
  def load_from_file(self, fname, param):
    with open(fname, 'r') as f:
      dict_spec = json.load(f)
      mod = Translator.from_spec(dict_spec, param)
    param.load(fname + '.data')
    return mod
