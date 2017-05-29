class ModelParams:
  """
  A structure that can be used to serialize the model and thus help with saving
  and loading the model
  """
  def __init__(self, encoder, attender, decoder, source_vocab, target_vocab,
               input_embedder, output_embedder):
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder
    self.source_vocab = source_vocab
    self.target_vocab = target_vocab
    self.input_embedder = input_embedder
    self.output_embedder = output_embedder
    self.serialize_params = [encoder, attender, decoder, source_vocab, target_vocab,
                             input_embedder, output_embedder]
