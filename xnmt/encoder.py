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
    return ExpressionSequence(expr_list=self.builder.transduce(sent))

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
    for module in self.modules:
      sent = module.transduce(sent)
    return sent

  def get_train_test_components(self):
    return self.modules

class SpeechBuilder(Encoder, Serializable):
  yaml_tag = u'!SpeechBuilder'
  def __init__(self, filter_height, filter_width, channels, num_filters, stride):
    """
    :param num_layers: depth of the RNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
    model = model_globals.get("dynet_param_collection").param_col  
    self.filter_height = filter_height
    self.filter_width = filter_width
    self.channels = channels
    self.num_filters = num_filters
    self.stride = stride # (2,2)
    
    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters1 = model.add_parameters(dim=(self.filter_height[0], self.filter_width[0], self.channels[0], self.num_filters[0]),
                                         init=normalInit)
    self.filters2 = model.add_parameters(dim=(self.filter_height[1], self.filter_width[1], self.channels[1], self.num_filters[1]),
                                         init=normalInit)
    self.filters3 = model.add_parameters(dim=(self.filter_height[2], self.filter_width[2], self.channels[2], self.num_filters[2]),
                                         init=normalInit)

  def whoami(self): return "Conv2dBuilder"

  def transduce(self, src):
    src = src.as_tensor()

    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    src_channels = 1
    batch_size = src.dim()[1]
    
    
    src = dy.reshape(src, (src_height, src_width, src_channels), batch_size=batch_size) # ((276, 80, 3), 1)
    # print(self.filters1)
    # convolution and pooling layers    
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filters1), stride = [self.stride[0], self.stride[0]], is_valid = True))
    pool1 = dy.maxpooling2d(l1, (1, 4), (1,2), is_valid = True)
    
    l2 = dy.rectify(dy.conv2d(pool1, dy.parameter(self.filters2), stride = [self.stride[1], self.stride[1]], is_valid = True))
    pool2 = dy.maxpooling2d(l2, (1, 4), (1,2), is_valid = True)

    l3 = dy.rectify(dy.conv2d(pool2, dy.parameter(self.filters3), stride = [self.stride[2], self.stride[2]], is_valid = True))

    pool3 = dy.kmax_pooling(l3, 1, d = 1)
    # print(pool3.dim())
    output = dy.cdiv(pool3,dy.sqrt(dy.squared_norm(pool3)))
    output = dy.reshape(output, (self.num_filters[2],), batch_size = batch_size)
    # print("my dim: ", output.dim())
    
    return ExpressionSequence(expr_tensor=output)

  def initial_state(self):
    return PseudoState(self)

class vgg16Builder(Encoder, Serializable):
  yaml_tag = u'!vgg16Builder'
  """
    Inputs are first put through 2 CNN layers, each with stride (2,2), so dimensionality
    is reduced by 4 in both directions.
    Then, we add a configurable number of bidirectional RNN layers on top.
    """
  
  def __init__(self, in_height, out_height):
    """
      :param num_layers: depth of the RNN
      :param input_dim: size of the inputs
      :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
      :param model
      :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
      """
    
    model = model_globals.get("dynet_param_collection").param_col 
    self.in_height = in_height
    self.out_height = out_height
    
    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)
  def whoami(self): return "vgg16Encoder"
  
  def transduce(self, src):
    src = src.as_tensor()
    
    src_height = src.dim()[0][0]
    src_width = 1
    batch_size = src.dim()[1]
    
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    
    src = dy.reshape(src, (src_height, src_width), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers
    l1 = (W*src)+b
    output = dy.cdiv(l1,dy.sqrt(dy.squared_norm(l1)))
    return ExpressionSequence(expr_tensor=output)
  
  def initial_state(self):
    return PseudoState(self)

