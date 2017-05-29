class ModelParams:
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  def __init__(self, encoder, attender, decoder, src_vocab, trg_vocab,
               input_embedder, output_embedder):
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.input_embedder = input_embedder
    self.output_embedder = output_embedder
    self.serialize_params = [encoder, attender, decoder, src_vocab, trg_vocab,
                             input_embedder, output_embedder]
