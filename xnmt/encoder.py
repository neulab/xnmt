import dynet as dy
import model
import embedder
import serializer
import model_globals

from decorators import recursive

# All types of encoder
import residual
import pyramidal
import conv_encoder

# Shortcut
Serializable = serializer.Serializable
HierarchicalModel = model.HierarchicalModel

class Encoder(HierarchicalModel):
  """
  An Encoder is a class that takes an ExpressionSequence as input and outputs another encoded ExpressionSequence.
  """
  def transduce(self, sent):
    """Encode inputs representing a sequence of continuous vectors into outputs that also represent a sequence of continuous vectors.

    :param sent: The input to be encoded. In the great majority of cases this will be an ExpressionSequence.
      It can be something else if the encoder is over something that is not a sequence of vectors though.
    :returns: The encoded output. In the great majority of cases this will be an ExpressionSequence.
      It can be something else if the encoder is over something that is not a sequence of vectors though.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return embedder.ExpressionSequence(expr_list=self.builder.transduce(sent))

class LSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!LSTMEncoder'

  def __init__(self, input_dim=None, layers=1, hidden_dim=None, dropout=None, bidirectional=True):
    super(LSTMEncoder, self).__init__()
    model = model_globals.dynet_param_collection.param_col
    input_dim = input_dim or model_globals.get("default_layer_dim")
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.input_dim = input_dim
    self.layers = layers
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    if bidirectional:
      self.builder = dy.BiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder)
    else:
      self.builder = dy.VanillaLSTMBuilder(layers, input_dim, hidden_dim, model)

  @recursive
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ResidualLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!ResidualLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None, bidirectional=True):
    super(ResidualLSTMEncoder, self).__init__()
    model = model_globals.dynet_param_collection.param_col
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    if bidirectional:
      self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    else:
      self.builder = residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)

  @recursive
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class PyramidalLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!PyramidalLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, downsampling_method="skip", reduce_factor=2, dropout=None):
    super(PyramidalLSTMEncoder, self).__init__()
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model_globals.dynet_param_collection.param_col, dy.VanillaLSTMBuilder, downsampling_method, reduce_factor)

  @recursive
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder, Serializable):
  yaml_tag = u'!ConvBiRNNBuilder'
  def init_builder(self, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    super(ConvBiRNNBuilder, self).__init__()
    model = model_globals.dynet_param_collection.param_col
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder,
                                            chn_dim, num_filters, filter_size_time, filter_size_freq, stride)

  @recursive
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ModularEncoder(Encoder, Serializable):
  yaml_tag = u'!ModularEncoder'
  def __init__(self, input_dim, modules):
    super(ModularEncoder, self).__init__()
    self.modules = modules

  def shared_params(self):
    return [set(["input_dim", "modules.0.input_dim"])]

  def transduce(self, sent):
    for module in self.modules:
      sent = module.transduce(sent)
    return sent

  @recursive
  def set_train(self, val):
    for module in self.modules:
      module.set_train(val)

