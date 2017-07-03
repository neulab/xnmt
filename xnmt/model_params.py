from yaml_serializer import Serializable, PostInitSharedParam

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, corpus_parser, translator):
    self.corpus_parser = corpus_parser
    self.translator = translator
    self.serialize_params = {"corpus_parser": corpus_parser, "translator": translator}
