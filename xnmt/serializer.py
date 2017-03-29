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
      json.dump(self.__to_spec(mod), f)
    params.save_all(fname + '.data')

  '''
  Load a model from a file.

  fname -- the file name
  spec -- the root model class
  param -- the model parameters
  '''
  def load_from_file(self, fname, param):
    with open(fname, 'r') as f:
      dict_spec = json.load(f)
      mod = self.__from_spec(dict_spec, param)
    param.load_all(fname + '.data')
    return mod

  def __to_spec(self, obj):
    print ("starting %s" % obj.__class__.__name__)
    if type(obj) == int or type(obj) == str or type(obj) == float:
      return obj
    info = {}
    info['__class__'] = obj.__class__.__name__
    if hasattr(obj, 'serialize_params'):
      info['__param__'] = [self.__to_spec(x) for x in obj.serialize_params]
    elif obj.__class__.__name__ != 'Model':
      raise NotImplementedError("Class %s is not serializable. Try adding serialize_params to it." % obj.__class__.__name__)
    print ("ending %s -- %r" % (obj.__class__.__name__, info))
    return info

  def __from_spec(self, spec, params):
    if type(spec) == int or type(spec) == str or type(spec) == float:
      return spec
    if type(spec) != dict:
      raise NotImplementedError("Class %s is not serializable. Try adding serialize_params to it." % spec.__class__.__name__)
    if not '__class__' in spec or not '__params__' in spec:
      raise NotImplementedError("Class %s is not serializable. Try adding __class__ and __params__ when saving it." % spec.__class__.__name__)
    args = [__from_spec(x, params) for x in spec['__params__']]
    return __import__(spec['__class__'])(**args)

    
