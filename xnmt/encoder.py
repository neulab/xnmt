import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder
from embedder import NoopEmbedder

class Encoder:
  '''
  A template class to encode an input.
  '''

  '''
  Takes an Input and returns an EncodedInput.
  '''

  def encode(self, x):
    raise NotImplementedError('encode must be implemented in Encoder subclasses')

  @staticmethod
  def from_spec(spec, encoder_layers, encoder_hidden_dim, input_embedder, model, residual_to_output):
    spec_lower = spec.lower()
    if spec_lower == "bilstm":
      return BiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model)
    elif spec_lower == "residuallstm":
      return ResidualLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model, residual_to_output)
    elif spec_lower == "residualbilstm":
      return ResidualBiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model, residual_to_output)
    elif spec_lower == "pyramidalbilstm":
      return PyramidalBiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model)
    elif spec_lower == "convbilstm":
      return ConvBiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model)
    elif spec_lower == "modular":
      return ModularEncoder([PyramidalBiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model),
                             BiLSTMEncoder(encoder_layers, encoder_hidden_dim, NoopEmbedder(encoder_hidden_dim, model), model)],
                            model
                            )
    else:
      raise RuntimeError("Unknown encoder type {}".format(spec_lower))


class DefaultEncoder(Encoder):

  def encode(self, sentence):
    embeddings = self.embedder.embed_sentence(sentence)
    return self.encoder.transduce(embeddings)


class BiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, output_dim, embedder, model]

class ResidualLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model, residual_to_output=False):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [layers, output_dim, embedder, model]

class ResidualBiLSTMEncoder(DefaultEncoder):
  """
  Implements a residual encoder with bidirectional first layer
  """

  def __init__(self, layers, output_dim, embedder, model, residual_to_output=False):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = residual.ResidualBiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder,
                                                 residual_to_output)
    self.serialize_params = [layers, output_dim, embedder, model]

class PyramidalBiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = pyramidal.PyramidalRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, output_dim, embedder, model]

class ConvBiLSTMEncoder(DefaultEncoder):

  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
    self.serialize_params = [layers, output_dim, embedder, model]

class ModularEncoder(Encoder):
  def __init__(self, module_list, model):
    self.module_list = module_list
    self.serialize_params = [model, ]

  def encode(self, sentence):
    for module in self.module_list:
      sentence = module.encode(sentence)
    return sentence
