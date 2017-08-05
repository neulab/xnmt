import dynet as dy
import expression_sequence
import model
import embedder
import serializer
import model_globals

from decorators import recursive
from expression_sequence import ExpressionSequence
from reports import HTMLReportable

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

  def calc_reinforce_loss(self, reward):
    return None

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

    return ExpressionSequence(expr_list=out)

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
                                                 model_globals.dynet_param_collection.param_col, dy.VanillaLSTMBuilder,
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

  def transduce(self, embed_sent):
    for module in self.modules:
      sent = module.transduce(embed_sent)
    return sent

  def get_train_test_components(self):
    return self.modules

class SegmentingEncoder(Encoder, Serializable, HTMLReportable):
  yaml_tag = u'!SegmentingEncoder'

  def __init__(self, embed_encoder=None, segment_transducer=None, lmbd=None, learn_segmentation=True):
    super(SegmentingEncoder, self).__init__()
    model = model_globals.dynet_param_collection.param_col

    self.ctr = 0
    self.lmbd_val = lmbd["start"]
    self.lmbd     = lmbd
    self.builder = segmenting_encoder.SegmentingEncoderBuilder(embed_encoder, segment_transducer,
                                                               learn_segmentation, model)

    self.register_hier_child(self.builder)

  def transduce(self, embed_sent):
    return ExpressionSequence(expr_tensor=self.builder.transduce(embed_sent))

  def calc_reinforce_loss(self, reward):
    return self.builder.calc_reinforce_loss(reward, self.lmbd_val)

  @recursive
  def set_train(self, val):
    pass

  def new_epoch(self):
    self.ctr += 1
#    self.lmbd_val *= self.lmbd["multiplier"]
    self.lmbd_val = 1e-3 * (2 * (2 ** (self.ctr-self.lmbd["before"]) -1))
    self.lmbd_val = min(self.lmbd_val, self.lmbd["max"])
    self.lmbd_val = max(self.lmbd_val, self.lmbd["min"])

    print("Now lambda:", self.lmbd_val)

class FullyConnectedEncoder(Encoder, Serializable):
  yaml_tag = u'!FullyConnectedEncoder'
  """
    Inputs are first put through 2 CNN layers, each with stride (2,2), so dimensionality
    is reduced by 4 in both directions.
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

  def transduce(self, embed_sent):
    src = embed_sent.as_tensor()
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
