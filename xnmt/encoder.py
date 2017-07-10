import dynet as dy
import residual
import pyramidal
import conv_encoder
#from speechEncoder import speechBuilder
#from vgg16Encoder import vgg16Builder
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
    model = model_globals.get("model")
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
    model = model_globals.get("model")
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
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model_globals.get("model"), dy.VanillaLSTMBuilder, downsampling_method, reduce_factor)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder, Serializable):
  yaml_tag = u'!ConvBiRNNBuilder'
  def init_builder(self, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    model = model_globals.get("model")
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

class ConvEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!ConvEncoder'
  def init_builder(self, chn_dims, filter_sizes, ksizes):
    self.filters = []
    self.biases = []
    self.layers = len(chn_dims)
    self.ksizes = ksizes
    for i in range(layers - 1):
      cur_w = dy.random_normal(tuple(filter_sizes[i] + [chn_dims[i], chn_dims[i + 1]]))
      cur_b = dy.random_normal((chn_dims[i + 1],))
      self.filters.append(cur_w)
      self.biases.append(cur_b)
    self.filters = tuple(self.filters)
    self.biases = tuple(self.biases)

  def transduce(self, sent):
    out_sent = []
    for x in sent:
      cur_a = x
      for i in range(self.layers - 1):
        cur_a = dy.conv2d_bias(cur_a, (self.filters[i]), (self.biases[i]), (1, 1), is_valid=False)
        cur_h = dy.rectify(cur_a)
        h_pool = dy.maxpooling2d(cur_h, (1, self.ksizes[0]), (1, 1), d=1)
      cur_sent = dy.kmax_pooling(h_pool, 1, d=1)
      out_sent.append(cur_sent)
    return out_sent

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
    #src = src.as_tensor()

    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    src_channels = src.dim()[0][2]
    batch_size = src.dim()[1]
    
    
    src = dy.reshape(src, (src_height, src_width, src_channels), batch_size=batch_size) # ((276, 80, 3), 1)
    print(self.filters1)
    # convolution and pooling layers    
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filters1), stride = [self.stride[0], self.stride[0]], is_valid = True))
    pool1 = dy.maxpooling2d(l1, (1, 4), (1,2), is_valid = True)
    
    l2 = dy.rectify(dy.conv2d(pool1, dy.parameter(self.filters2), stride = [self.stride[1], self.stride[1]], is_valid = True))
    pool2 = dy.maxpooling2d(l2, (1, 4), (1,2), is_valid = True)

    l3 = dy.rectify(dy.conv2d(pool2, dy.parameter(self.filters3), stride = [self.stride[2], self.stride[2]], is_valid = True))

    pool3 = dy.kmax_pooling(l3, 1, d = 1)
    print(pool3.dim())
    output = dy.cdiv(pool3,dy.sqrt(dy.squared_norm(pool3)))
    
    return output

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
    #src = src.as_tensor()
    
    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    batch_size = src.dim()[1]
    
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    
    src = dy.reshape(src, (src_height, src_width), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers
    l1 = (W*src)+b
    output = dy.cdiv(l1,dy.sqrt(dy.squared_norm(l1)))
    return output
  
  def initial_state(self):
    return PseudoState(self)


