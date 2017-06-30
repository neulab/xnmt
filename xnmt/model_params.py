from yaml_serializer import Serializable, PostInitSharedParam

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, translator, src_reader, trg_reader):
    self.translator = translator
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.serialize_params = {"translator": self.translator,
                             "src_reader": self.src_reader,
                             "trg_reader": self.trg_reader}
  def shared_params_post_init(self):
    return [
            PostInitSharedParam(model="translator.encoder", param="vocab_size", value=lambda: len(self.src_reader.vocab)),
            PostInitSharedParam(model="translator.decoder", param="vocab_size", value=lambda: len(self.trg_reader.vocab)),
            PostInitSharedParam(model="translator.output_embedder", param="vocab_size", value=lambda: len(self.trg_reader.vocab)),
            ]
