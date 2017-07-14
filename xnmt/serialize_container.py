from serializer import Serializable

class SerializeContainer(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """

  yaml_tag = "!SerializeContainer"
  def __init__(self, corpus_parser, model, model_globals):
    self.corpus_parser = corpus_parser
    self.model = model
    self.model_globals = model_globals
    self.serialize_params = {"corpus_parser": corpus_parser, "model":model, "model_globals": model_globals}
