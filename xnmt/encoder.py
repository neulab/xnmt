import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder

class Encoder:
  """
  A parent class representing all classes that encode inputs.
  """

  def transduce(self, sentence):
    """
    Encode inputs into outputs.
    :param sentence: The input to be encoded. This is duck-typed, so it is the
      appropriate input for this particular type of encoder. Frequently it will
      be a list of word embeddings, but it can be anything else.
    :returns: The encoded output. Frequently this will be a list of expressions
      representing the encoded vectors for each word.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')

  @staticmethod
  def from_spec(spec, layers, input_dim, output_dim, model, residual_to_output):
    spec_lower = spec.lower()
    if spec_lower == "bilstm":
      return BiRNNEncoder(layers, input_dim, output_dim, model)
    elif spec_lower == "residuallstm":
      return residual.ResidualRNNBuilder(layers, input_dim, input_dim, output_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    elif spec_lower == "residualbilstm":
      return residual.ResidualBiRNNBuilder(layers, input_dim, input_dim, output_dim, model, dy.VanillaLSTMBuilder,
                                                 residual_to_output)
    elif spec_lower == "pyramidalbilstm":
      return pyramidal.PyramidalRNNBuilder(layers, input_dim, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    elif spec_lower == "convbilstm":
      return conv_encoder.ConvBiRNNBuilder(layers, input_dim, input_dim, output_dim, model, dy.LSTMBuilder)
    elif spec_lower == "modular":
      # example for a modular encoder: stacked pyramidal encoder, followed by stacked LSTM 
      return ModularEncoder([
                             pyramidal.PyramidalRNNBuilder(layers, input_dim, input_dim, output_dim, model, dy.VanillaLSTMBuilder),
                             dy.BiRNNBuilder(layers, output_dim, output_dim, model, dy.VanillaLSTMBuilder)
                             ],
                            model
                            )
    else:
      raise RuntimeError("Unknown encoder type {}".format(spec_lower))

class BiRNNEncoder(Encoder):
  def __init__(self, layers, input_dim, output_dim, model):
    self.builder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, input_dim, output_dim, model]

  def transduce(self, sentence):
    return self.builder.transduce(sentence)

class ModularEncoder(Encoder):
  def __init__(self, module_list, model):
    self.module_list = module_list
    self.serialize_params = [model, ]
    self.embedder = module_list[0].embedder

  def transduce(self, sentence):
    for i, module in enumerate(self.module_list):
      sentence = module.transduce(sentence)
      if i<len(self.module_list)-1:
        sentence = ExpressionSequence(expr_list=sentence)
    return sentence
