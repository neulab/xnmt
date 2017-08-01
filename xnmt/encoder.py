import dynet as dy
import expression_sequence
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

class FullyConnectedEncoder(Encoder, Serializable):
  yaml_tag = u'!FullyConnectedEncoder'
  """
    Then, we add a configurable number of bidirectional RNN layers on top.
    """

  def __init__(self, in_height, out_height, nonlinearity='linear'):
    """
      :param num_layers: depth of the RNN
      :param input_dim: size of the inputs
      :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
      :param model
      :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
      """

    model = model_globals.dynet_param_collection.param_col
    self.in_height = in_height
    self.out_height = out_height
    self.nonlinearity = nonlinearity

    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)

  def transduce(self, src):
    src = src.as_tensor()
    src_height = src.dim()[0][0]
    src_width = 1
    batch_size = src.dim()[1]

    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)

    #src = dy.reshape(src, (src_height, src_width), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers
    #l1 = (W*src)+b
    l1 = dy.affine_transform([b, W, src])
    output = l1
    if self.nonlinearity is 'linear':
      output = l1
    else:
      if self.nonlinearity is 'sigmoid':
        output = dy.logistic(l1)
      else:
        if self.nonlinearity is 'tanh':
          output = 2*dy.logistic(l1) - 1
        else:
          if self.nonlinearity is 'relu':
            output = dy.rectify(l1)
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
