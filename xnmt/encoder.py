import dynet as dy
from batcher import *
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


class DefaultEncoder(Encoder):

  def encode(self, sentence):
    embeddings = self.embedder.embed_sentence(sentence)
    return self.encoder.transduce(embeddings)


class BiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
    self.model_lookup = model.add_lookup_parameters((embedder.vocab_size, embedder.emb_dim))


class ResidualLSTMEncoder(DefaultEncoder):
  
  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
