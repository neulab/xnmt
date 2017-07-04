from serializer import Serializable

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, corpus_parser, model, global_params):
    self.corpus_parser = corpus_parser
    self.model = model
    self.global_params = global_params
    self.serialize_params = {"corpus_parser": corpus_parser, "model":model, "global_params": global_params}
