from __future__ import print_function

import sys
import dynet as dy
import expression_sequence
import model
import embedder
import serializer
import model_globals
import pdb
from decorators import recursive
from expression_sequence import ExpressionSequence
from reports import Reportable

# The LSTM model builders
import pyramidal
import conv_encoder
import residual
import segmenting_encoder

# Shortcut
Serializable = serializer.Serializable
HierarchicalModel = model.HierarchicalModel

class Encoder(HierarchicalModel):
  """
  An Encoder is a class that takes an ExpressionSequence as input and outputs another encoded ExpressionSequence.
  """
  def transduce(self, embed_sent):
    """Encode inputs representing a sequence of continuous vectors into outputs that also represent a sequence of continuous vectors.

    :param sent: The input to be encoded. In the great majority of cases this will be an ExpressionSequence.
      It can be something else if the encoder is over something that is not a sequence of vectors though.
    :returns: The encoded output. In the great majority of cases this will be an ExpressionSequence.
      It can be something else if the encoder is over something that is not a sequence of vectors though.
    """
    raise NotImplementedError('Unimplemented transduce for class:', self.__class__.__name__)

class BuilderEncoder(Encoder):
  def transduce(self, embed_sent):
    out = None
    if hasattr(self.builder, "transduce"):
      out = self.builder.transduce(embed_sent)
    elif hasattr(self.builder, "initial_state"):
      out = self.builder.initial_state().transduce(embed_sent)
    else:
      raise NotImplementedError("Unimplemented transduce logic for class:",
                                self.builder.__class__.__name__)
    return ExpressionSequence(expr_list=out, mask=embed_sent.mask)

class IdentityEncoder(Encoder, Serializable):
  yaml_tag = u'!IdentityEncoder'

  def transduce(self, embed_sent):
    return embed_sent

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
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim,
                                                 model_globals.dynet_param_collection.param_col, dy.CompactVanillaLSTMBuilder,
                                                 downsampling_method, reduce_factor)
 
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
                                                 chn_dim, num_filters, filter_size_time, filter_size_freq,
                                                 stride)

  @recursive
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ModularEncoder(Encoder, Serializable):
  yaml_tag = u'!ModularEncoder'

  def __init__(self, input_dim, modules):
    super(ModularEncoder, self).__init__()
    self.modules = modules

    for module in self.modules:
      self.register_hier_child(module)

  def shared_params(self):
    return [set(["input_dim", "modules.0.input_dim"])]

  def transduce(self, sent):
    for module in self.modules:
      sent = module.transduce(sent)
    return sent

  def get_train_test_components(self):
    return self.modules

class SegmentingEncoder(Encoder, Serializable, Reportable):
  yaml_tag = u'!SegmentingEncoder'

  def __init__(self, embed_encoder=None, segment_transducer=None, lmbd_learning=None, learn_segmentation=True):
    super(SegmentingEncoder, self).__init__()
    model = model_globals.dynet_param_collection.param_col

    self.ctr = 0
    self.lmbd     = lmbd_learning["initial"]
    self.lmbd_max = lmbd_learning["max"]
    self.lmbd_min = lmbd_learning["min"]
    self.lmbd_grw = lmbd_learning["grow"]
    self.warmup   = lmbd_learning["warmup"]
    self.builder = segmenting_encoder.SegmentingEncoderBuilder(embed_encoder, segment_transducer,
                                                               learn_segmentation, model)

    self.register_hier_child(self.builder)

  def transduce(self, embed_sent):
    lmbd = 0 if self.ctr < self.warmup else self.lmbd
    return ExpressionSequence(expr_tensor=self.builder.transduce(embed_sent, lmbd))

  def calc_additional_loss(self, reward):
    lmbd = 0 if self.ctr < self.warmup else self.lmbd
    return self.builder.calc_additional_loss(reward, lmbd)

  @recursive
  def set_train(self, val):
    pass

  def new_epoch(self):
    self.ctr += 1

    if self.ctr > self.warmup:
      self.lmbd *= self.lmbd_grw
      self.lmbd = min(self.lmbd, self.lmbd_max)
      self.lmbd = max(self.lmbd, self.lmbd_min)
      print("Now lambda:", self.lmbd, file=sys.stderr)

class FullyConnectedEncoder(Encoder, Serializable):
  yaml_tag = u'!FullyConnectedEncoder'
  def __init__(self, in_height, out_height, nonlinearity='linear'):
    """
      :param in_height, out_height: input and output dimension of the affine transform 
      :param nonlinearity: nonlinear activation function
    """
    model = model_globals.dynet_param_collection.param_col
    self.in_height = in_height
    self.out_height = out_height
    self.nonlinearity = nonlinearity

    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)

  def transduce(self, embed_sent):
    src = embed_sent.as_tensor()
    src_height = src.dim()[0][0]
    src_width = 1
    batch_size = src.dim()[1]

    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
 
    l1 = dy.affine_transform([b, W, src])
    output = l1
    if self.nonlinearity is 'linear':
      output = l1
    elif  self.nonlinearity is 'sigmoid':
      output = dy.logistic(l1)
    elif self.nonlinearity is 'tanh':
      output = 2*dy.logistic(l1) - 1
    elif self.nonlinearity is 'relu':
      output = dy.rectify(l1)
    return expression_sequence.ExpressionSequence(expr_tensor=output)

  def initial_state(self):
    return PseudoState(self)


class ConvConnectedEncoder(Encoder, Serializable):
  yaml_tag = u'!ConvConnectedEncoder'
  """
    Input goes through through a first convolution in time and space, no stride, 
    dimension is not reduced, then CNN layer for each frame several times
    Embedding sequence has same length as Input sequence
    """

  def __init__(self, input_dim, window_receptor,output_dim,num_layers,internal_dim,non_linearity='linear'):
    """
      :param num_layers: num layers after first receptor conv
      :param input_dim: size of the inputs
      :param window_receptor: window for the receptor 
      :param ouput_dim: size of the outputs
      :param internal_dim: size of hidden dimension, internal dimension
      :param non_linearity: Non linearity to apply between layers
      """

    model = model_globals.dynet_param_collection.param_col
    self.input_dim = input_dim
    self.window_receptor = window_receptor
    self.internal_dim = internal_dim
    self.non_linearity = non_linearity
    self.output_dim = output_dim
    if self.non_linearity == 'linear':
        self.gain = 1.0
    elif self.non_linearity == 'tanh':
        self.gain = 1.0
    elif self.non_linearity == 'relu':
        self.gain = 0.5
    elif self.non_linearity == 'sigmoid':
        self.gain = 4.0

    glorotInit=dy.GlorotInitializer(is_lookup=False, gain=self.gain)
    normalInit=dy.NormalInitializer(0, 0.1)
     
    self.pConv1 = model.add_parameters(dim = (self.input_dim,self.window_receptor,1,self.internal_dim),init=normalInit)
    self.pBias1 = model.add_parameters(dim = (self.internal_dim))
    self.builder_layers = []
    for index in range(num_layers):
        conv = model.add_parameters(dim = (self.internal_dim,1,1,self.internal_dim),init=normalInit)
        bias = model.add_parameters(dim = (self.internal_dim))
        self.builder_layers.append((conv,bias))

    self.last_conv = model.add_parameters(dim = (self.internal_dim,1,1,self.output_dim),init=normalInit)
    self.last_bias = model.add_parameters(dim = (self.output_dim))

  def whoami(self): return "ConvConnectedEncoder"

  def transduce(self, embed_sent):
    src = embed_sent.as_tensor()
    
    sent_len = src.dim()[0][1]
    src_width = 1
    batch_size = src.dim()[1]
    pad_size = (self.window_receptor-1)/2 #TODO adapt it also for even window size

    src = dy.concatenate([dy.zeroes((self.input_dim,pad_size),batch_size=batch_size),src,dy.zeroes((self.input_dim,pad_size), batch_size=batch_size)],d=1)
    padded_sent_len = sent_len + 2*pad_size

    conv1 = dy.parameter(self.pConv1)
    bias1 = dy.parameter(self.pBias1)
    src_chn = dy.reshape(src,(self.input_dim,padded_sent_len,1),batch_size=batch_size)
    cnn_layer1 = dy.conv2d_bias(src_chn,conv1,bias1,stride=[1,1])
    
    hidden_layer = dy.reshape(cnn_layer1,(self.internal_dim,sent_len,1),batch_size=batch_size)
    if self.non_linearity is 'linear':
        hidden_layer = hidden_layer
    elif self.non_linearity is 'tanh':
        hidden_layer = dy.tanh(hidden_layer)
    elif self.non_linearity is 'relu':
        hidden_layer = dy.rectify(hidden_layer)
    elif self.non_linearity is 'sigmoid':
        hidden_layer = dy.logistic(hidden_layer)
    
    for conv_hid, bias_hid in self.builder_layers:
        hidden_layer = dy.conv2d_bias(hidden_layer, dy.parameter(conv_hid),dy.parameter(bias_hid),stride=[1,1])
        hidden_layer = dy.reshape(hidden_layer,(self.internal_dim,sent_len,1),batch_size=batch_size)
        if self.non_linearity is 'linear':
            hidden_layer = hidden_layer
        elif self.non_linearity is 'tanh':
            hidden_layer = dy.tanh(hidden_layer)
        elif self.non_linearity is 'relu':
            hidden_layer = dy.rectify(hidden_layer)
        elif self.non_linearity is 'sigmoid':
            hidden_layer = dy.logistic(hidden_layer)
    last_conv = dy.parameter(self.last_conv)
    last_bias = dy.parameter(self.last_bias)
    output = dy.conv2d_bias(hidden_layer,last_conv,last_bias,stride=[1,1])
    #output = dy.reshape(output, (self.output_dim,sent_len),batch_size=batch_size)
    output = dy.reshape(output, (sent_len,self.output_dim),batch_size=batch_size)
    return expression_sequence.ExpressionSequence(expr_tensor=output)
  
  def initial_state(self):
    return PseudoState(self)



if __name__ == '__main__':
  # To use this code, comment out the model initialization in the class and the line for src.as_tensor()
  dy.renew_cg()
  model = dy.ParameterCollection()
  l1 = FullyConnectedEncoder(2, 1, 'sigmoid')
  a = dy.inputTensor([1, 2])
  b = l1.transduce(a)
  print(b[0].npvalue())

  l2 = FullyConnectedEncoder(2, 1, 'tanh')
  c = l2.transduce(a)
  print(c[0].npvalue())

  l3 = FullyConnectedEncoder(2, 1, 'linear')
  d = l3.transduce(a)
  print(d[0].npvalue())

  l4 = FullyConnectedEncoder(2, 1, 'relu')
  e = l4.transduce(a)
  print(e[0].npvalue())




