from yaml_serializer import Serializable

class ModelParams(Serializable):
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  
  yaml_tag = "!ModelParams"
  def __init__(self, encoder, attender, decoder, src_vocab, trg_vocab,
               input_embedder, output_embedder):
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.input_embedder = input_embedder
    self.output_embedder = output_embedder
    self.serialize_params = {"encoder": encoder, 
                             "attender": attender,
                             "decoder": decoder,
                             "src_vocab": src_vocab, 
                             "trg_vocab": trg_vocab,
                             "input_embedder": input_embedder,
                             "output_embedder": output_embedder}
