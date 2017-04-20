import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder

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
    self.serialize_params = [layers, output_dim, embedder, model]

class ResidualLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model, residual_to_output=False):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder, residual_to_output)
    self.serialize_params = [layers, output_dim, embedder, model]

class ResidualBiLSTMEncoder(DefaultEncoder):
  """
  Implements a residual encoder with bidirectional first layer
  """

  def __init__(self, layers, output_dim, embedder, model, residual_to_output=False):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualBiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder,
                                                 residual_to_output)
    self.serialize_params = [layers, output_dim, embedder, model]

class PyramidalBiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = pyramidal.PyramidalRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
    self.serialize_params = [layers, output_dim, embedder, model]

class ConvBiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
    self.serialize_params = [layers, output_dim, embedder, model]

