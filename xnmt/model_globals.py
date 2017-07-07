import dynet as dy

model_globals = {
          "dynet_param_collection" : None,
          "dropout" : 0.0,
          "default_layer_dim" : 512,
          }
get = model_globals.get # shortcut

class PersistentParamCollection(object):
  def __init__(self, model_filename):
    self.param_col = dy.Model()