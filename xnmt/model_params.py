from yaml_serializer import Serializable, PostInitSharedParam

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, src_reader, trg_reader, translator):
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.translator = translator
#    self.serialize_params = {"translator": self.translator,
#                             "src_reader": self.src_reader,
#                             "trg_reader": self.trg_reader}
  def shared_params(self):
    return [
            set(["src_reader.max_num_train_sents", "trg_reader.max_num_train_sents"]),
            set(["src_reader.max_num_dev_sents", "trg_reader.max_num_dev_sents"]),
            ]
  def shared_params_post_init(self):
    return [
            PostInitSharedParam(model="translator.input_embedder", param="vocab_size", value=self.get_src_vocab_size),
            PostInitSharedParam(model="translator.decoder", param="vocab_size", value=self.get_trg_vocab_size),
            PostInitSharedParam(model="translator.output_embedder", param="vocab_size", value=self.get_trg_vocab_size),
            ]
  def get_src_vocab_size(self):
    return len(self.init_params["src_reader"].vocab)
  def get_trg_vocab_size(self):
    return len(self.init_params["trg_reader"].vocab)
  