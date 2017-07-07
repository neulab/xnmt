import dynet as dy
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
from translator import TrainTestInterface
from serializer import Serializable
import model_globals

class Encoder(TrainTestInterface):
  """
  A parent class representing all classes that encode inputs.
  """
  def __init__(self, model, global_train_params, input_dim):
    """
    Every encoder constructor needs to accept at least these 3 parameters 
    """
    raise NotImplementedError('__init__ must be implemented in Encoder subclasses')

  def transduce(self, sent):
    """Encode inputs into outputs.

    :param sent: The input to be encoded. This is duck-typed, so it is the appropriate input for this particular type of encoder. Frequently it will be a list of word embeddings, but it can be anything else.
    :returns: The encoded output. Frequently this will be a list of expressions representing the encoded vectors for each word.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return self.builder.transduce(sent)

class LSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!LSTMEncoder'

  def __init__(self, input_dim=None, layers=1, hidden_dim=None, dropout=None, bidirectional=True):
    model = model_globals.get("dynet_param_collection").param_col
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
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)


class ResidualLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!ResidualLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None, bidirectional=True):
    model = model_globals.get("dynet_param_collection").param_col
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    if bidirectional:
      self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    else:
      self.builder = residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class PyramidalLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!PyramidalLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, downsampling_method="skip", reduce_factor=2, dropout=None):
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model_globals.get("dynet_param_collection").param_col, dy.VanillaLSTMBuilder, downsampling_method, reduce_factor)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder, Serializable):
  yaml_tag = u'!ConvBiRNNBuilder'
  def init_builder(self, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    model = model_globals.get("dynet_param_collection").param_col
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder,
                                            chn_dim, num_filters, filter_size_time, filter_size_freq, stride)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
  
class ModularEncoder(Encoder, Serializable):
  yaml_tag = u'!ModularEncoder'
  def __init__(self, input_dim, modules):
    self.modules = modules
    
  def shared_params(self):
    return [set(["input_dim", "modules.0.input_dim"])]

  def transduce(self, sent):
    for i, module in enumerate(self.modules):
      sent = module.transduce(sent)
      if i<len(self.modules)-1:
        sent = ExpressionSequence(expr_list=sent)
    return sent

  def get_train_test_components(self):
    return self.modules
