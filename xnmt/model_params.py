from serializer import Serializable

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, corpus_parser, model):
    self.corpus_parser = corpus_parser
    self.model = model
    # TODO: we need to store the model_globals params in here, as well
    self.serialize_params = {"corpus_parser": corpus_parser, "model": model}
