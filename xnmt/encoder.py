import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
from translator import TrainTestInterface
import inspect
from yaml_serializer import Serializable
import model_globals
import yaml

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

  @staticmethod
  def from_spec(encoder_spec, global_train_params, model):
    """Create an encoder from a specification.

    :param encoder_spec: Encoder-specific settings (encoders must consume all provided settings)
    :param global_train_params: dictionary with global params such as dropout and default_layer_dim, which the encoders are free to make use of.
    :param model: The model that we should add the parameters to
    """
    encoder_spec = dict(encoder_spec)
    encoder_type = encoder_spec.pop("type")
    encoder_spec["model"] = model
    encoder_spec["global_train_params"] = global_train_params
    known_encoders = [key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder) and key not in ["BuilderEncoder","Encoder"]]
    if encoder_type not in known_encoders and encoder_type+"Encoder" not in known_encoders:
      raise RuntimeError("specified encoder %s is unknown, choices are: %s" 
                         % (encoder_type,", ".join([key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder)])))
    encoder_class = globals().get(encoder_type, globals().get(encoder_type+"Encoder"))
    return encoder_class(**encoder_spec)

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return self.builder.transduce(sent)

class BiLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!BiLSTMEncoder'
  def __repr__(self):
    return "%s(input_dim=%r, layers=%r, hidden_dim=%r, dropout=%r)" % (
            self.__class__.__name__,
            self.input_dim, self.layers, self.hidden_dim, self.dropout)

  def __init__(self, input_dim=None, layers=1, hidden_dim=None, dropout=None):
    model = model_globals.model
    if input_dim is None: input_dim = model_globals.default_layer_dim
    if hidden_dim is None: hidden_dim = model_globals.default_layer_dim
    if dropout is None: dropout = model_globals.dropout
    self.input_dim = input_dim
    self.layers = layers
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.builder = dy.BiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)


class ResidualLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, residual_to_output, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ResidualBiLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, residual_to_output, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class PyramidalLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, downsampling_method="skip", dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, downsampling_method)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, downsampling_method, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder):
  def init_builder(self, model, global_train_params, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder,
                                            chn_dim, num_filters, filter_size_time, filter_size_freq, stride)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, chn_dim, num_filters, filter_size_time, filter_size_freq, stride, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
  
class ModularEncoder(Encoder):
  def __init__(self, model, global_train_params, input_dim, modules):
    self.modules = []
    if input_dim != modules[0].get("input_dim"):
      raise RuntimeError("Mismatching input dimensions of first module: %s != %s".format(input_dim, modules[0].get("input_dim")))
    for module_spec in modules:
      self.modules.append(Encoder.from_spec(module_spec, global_train_params, model))
    self.serialize_params = [model, global_train_params, input_dim, modules]

  def transduce(self, sent):
    for i, module in enumerate(self.modules):
      sent = module.transduce(sent)
      if i<len(self.modules)-1:
        sent = ExpressionSequence(expr_list=sent)
    return sent

  def get_train_test_components(self):
    return self.modules
