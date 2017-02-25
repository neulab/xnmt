import dynet as dy
import residual

class Encoder:
  '''
  A template class to encode an input.
  '''
  
  '''
  Takes an Input and returns an EncodedInput.
  '''
  def encode(self, x):
    raise NotImplementedError('encode must be implemented in Encoder subclasses')

class BiLstmEncoder(Encoder):
  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)

  def encode(self, sentence):
    embeddings = [self.embedder.embed(word) for word in sentence]
    return self.encoder.transduce(embeddings)


class ResidualLstmEncoder(Encoder):
  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)

  def encode(self, sentence):
    embeddings = [self.embedder.embed(word) for word in sentence]
    return self.encoder.transduce(embeddings)
