from __future__ import print_function
from embedder import *
from attender import *
from input import *
from encoder import *
from decoder import *
from translator import *
from serializer import *
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
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(fname, 'w') as f:
      json.dump(self.__to_spec(mod), f)
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
      mod = self.__from_spec(dict_spec, param)
    param.populate(fname + '.data')
    return mod

  def __to_spec(self, obj):
    if type(obj) == int or type(obj) == str or type(obj) == float or type(obj) == bool or type(obj)==unicode:
      return obj
    info = {}
    info['__class__'] = obj.__class__.__name__
    if hasattr(obj, 'serialize_params'):
      info['__module__'] = obj.__module__
      info['__param__'] = [self.__to_spec(x) for x in obj.serialize_params]
    elif obj.__class__.__name__ == 'list' or obj.__class__.__name__ == 'dict':
      return json.dumps(obj)
    elif obj.__class__.__name__ != 'ParameterCollection':
      raise NotImplementedError("Class %s is not serializable. Try adding serialize_params to it." % obj.__class__.__name__)
    return info

  def __from_spec(self, spec, params):
    if type(spec) == int or type(spec) == float or type(spec) == bool:
      return spec
    if type(spec) == unicode:
      try:
        return json.loads(spec)
      except ValueError:
        return spec
    if type(spec) != dict:
      raise NotImplementedError("Class %s is not deserializable. Try adding serialize_params to it." % spec.__class__.__name__)
    elif '__class__' not in spec:
      raise NotImplementedError("Dict is not deserializable. Try adding __class__ when saving it:\n %r" % spec)
    elif spec['__class__'] == 'ParameterCollection':
      return params
    elif '__param__' not in spec:
      raise NotImplementedError("Dict is not deserializable. Try adding __param__ when saving it:\n %r" % spec)
    args = [self.__from_spec(x, params) for x in spec['__param__']]
    _class = getattr(__import__(spec['__module__']), spec['__class__'])
    return _class(*args)
