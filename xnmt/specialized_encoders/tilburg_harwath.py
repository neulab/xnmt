import dynet as dy

from xnmt.expression_seqs import ExpressionSequence
from xnmt.param_collections import ParamManager
from xnmt.persistence import Serializable, serializable_init
from xnmt.transducers.base import SeqTransducer

# This is a file for specialized encoders that implement a particular model
# Ideally, these will eventually be refactored to use standard components and the ModularSeqTransducer framework,
#  (for more flexibility), but for ease of implementation it is no problem to perform an initial implementation here.

def padding(src, min_size):
  """ do padding for the sequence input along the time step (for example speech), so that so that the output of convolutional layer has the same size(time) of the input.

      note that for padding image(two dimensional padding), please refer to dyne.conv2d(..., is_valid = False)
  """
  # pad before put into convolutional layer
  src_dim = src.dim()
  if src_dim[0][1] >= min_size:
    return src
  pad_size = min_size - src_dim[0][1]
  channels = src_dim[0][2] if len(src_dim[0]) >= 3 else 1
  if pad_size == 1:
    return dy.concatenate([src, dy.zeroes((src_dim[0][0], 1, channels))], d=1)
  else:
    left_border = int(pad_size) / 2
    right_border = (int(pad_size)+1) / 2
    return dy.concatenate([dy.zeroes((src_dim[0][0], left_border, channels)), src, dy.zeroes((src_dim[0][0], right_border, channels))], d=1) # do concatenate along cols


class TilburgSpeechSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!TilburgSpeechSeqTransducer'

  @serializable_init
  def __init__(self, filter_height, filter_width, channels, num_filters, stride, rhn_num_hidden_layers, rhn_dim,
               rhn_microsteps, attention_dim, residual= False):
    self.filter_height = filter_height
    self.filter_width = filter_width
    self.channels = channels
    self.num_filters = num_filters
    self.stride = stride
    self.rhn_num_hidden_layers = rhn_num_hidden_layers
    self.rhn_dim = rhn_dim
    self.rhn_microsteps = rhn_microsteps
    self.attention_dim = attention_dim
    self.residual = residual

    model = ParamManager.my_params(self)
    # Convolutional layer
    self.filter_conv = model.add_parameters(dim=(self.filter_height, self.filter_width, self.channels, self.num_filters))
    # Recurrent highway layer
    self.recur  = []
    self.linear = []
    self.init   = []
    self.attention = []

    input_dim = num_filters
    for _ in range(rhn_num_hidden_layers):
      self.init.append(model.add_parameters((rhn_dim,)))
      self.linear.append((model.add_parameters((rhn_dim, input_dim)),
                          model.add_parameters((rhn_dim, input_dim,))))
      input_dim = rhn_dim
      recur_layer = []
      for _ in range(self.rhn_microsteps):
        recur_layer.append((model.add_parameters((rhn_dim, rhn_dim)),
                            model.add_parameters((rhn_dim,)),
                            model.add_parameters((rhn_dim, rhn_dim,)),
                            model.add_parameters((rhn_dim,))))
      self.recur.append(recur_layer)
    # Attention layer
    self.attention.append((model.add_parameters((attention_dim, rhn_dim)),
                           model.add_parameters(attention_dim, )))

  def transduce(self, src: ExpressionSequence) -> ExpressionSequence:
    src = src.as_tensor()
    # convolutional layer
    src = padding(src, src.dim()[0][0], src.dim()[0][1], self.filter_width, self.stride, src.dim()[1])
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filter_conv), stride = [self.stride, self.stride], is_valid = True))
    timestep = l1.dim()[0][1]
    features = l1.dim()[0][2]
    batch_size = l1.dim()[1]
    # transpose l1 to be (timesetp, dim), but keep the batch_size.
    rhn_in = dy.reshape(l1, (timestep, features), batch_size = batch_size)
    rhn_in = [dy.pick(rhn_in, i) for i in range(timestep)]
    for l in range(self.rhn_num_hidden_layers):
      rhn_out = []
      # initialize a random vector for the first state vector, keep the same batch size.
      prev_state = dy.parameter(self.init[l])
      # begin recurrent high way network
      for t in range(timestep):
        for m in range(0, self.rhn_microsteps):
          H = dy.affine_transform([dy.parameter(self.recur[l][m][1]), dy.parameter(self.recur[l][m][0]),  prev_state])
          T = dy.affine_transform([dy.parameter(self.recur[l][m][3]), dy.parameter(self.recur[l][m][2]),  prev_state])
          if m == 0:
            H += dy.parameter(self.linear[l][0]) * rhn_in[t]
            T += dy.parameter(self.linear[l][1]) * rhn_in[t]
          H = dy.tanh(H)
          T = dy.logistic(T)
          prev_state = dy.cmult(1 - T, prev_state) + dy.cmult(T, H) # ((1024, ), batch_size)
        rhn_out.append(prev_state)
      if self.residual and l>0:
        rhn_out = [sum(x) for x in zip(rhn_out, rhn_in)]
      rhn_in = rhn_out
    # Compute the attention-weighted average of the activations
    rhn_in = dy.concatenate_cols(rhn_in)
    scores = dy.transpose(dy.parameter(self.attention[0][1]))*dy.tanh(dy.parameter(self.attention[0][0])*rhn_in) # ((1,510), batch_size)
    scores = dy.reshape(scores, (scores.dim()[0][1],), batch_size = scores.dim()[1])
    attn_out = rhn_in*dy.softmax(scores) # # rhn_in.as_tensor() is ((1024,510), batch_size) softmax is ((510,), batch_size)
    return ExpressionSequence(expr_tensor = attn_out)

# This is a CNN-based encoder that was used in the following paper:
#  http://papers.nips.cc/paper/6186-unsupervised-learning-of-spoken-language-with-visual-context.pdf
class HarwathSpeechSeqTransducer(SeqTransducer, Serializable):
  yaml_tag = '!HarwathSpeechSeqTransducer'

  @serializable_init
  def __init__(self, filter_height, filter_width, channels, num_filters, stride):
    """
    Args:
      num_layers: depth of the RNN
      input_dim: size of the inputs
      hidden_dim: size of the outputs (and intermediate RNN layer representations)
    """
    model = ParamManager.my_params(self)
    self.filter_height = filter_height
    self.filter_width = filter_width
    self.channels = channels
    self.num_filters = num_filters
    self.stride = stride # (2,2)
    self.hidden_states = {}

    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters1 = model.add_parameters(dim=(self.filter_height[0], self.filter_width[0], self.channels[0], self.num_filters[0]),
                                         init=normalInit)
    self.filters2 = model.add_parameters(dim=(self.filter_height[1], self.filter_width[1], self.channels[1], self.num_filters[1]),
                                         init=normalInit)
    self.filters3 = model.add_parameters(dim=(self.filter_height[2], self.filter_width[2], self.channels[2], self.num_filters[2]),
                                         init=normalInit)

  def transduce(self, src: ExpressionSequence) -> ExpressionSequence:
    src = src.as_tensor()

    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    # src_channels = 1
    batch_size = src.dim()[1]

    # convolution and pooling layers
    # src dim is ((40, 1000), 128)
    src = padding(src, self.filter_width[0]+3)
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filters1), stride = [self.stride[0], self.stride[0]], is_valid = True)) # ((1, 1000, 64), 128)
    pool1 = dy.maxpooling2d(l1, (1, 4), (1,2), is_valid = True) #((1, 499, 64), 128)

    pool1 = padding(pool1, self.filter_width[1]+3)
    l2 = dy.rectify(dy.conv2d(pool1, dy.parameter(self.filters2), stride = [self.stride[1], self.stride[1]], is_valid = True))# ((1, 499, 512), 128)
    pool2 = dy.maxpooling2d(l2, (1, 4), (1,2), is_valid = True)#((1, 248, 512), 128)

    pool2 = padding(pool2, self.filter_width[2])
    l3 = dy.rectify(dy.conv2d(pool2, dy.parameter(self.filters3), stride = [self.stride[2], self.stride[2]], is_valid = True))# ((1, 248, 1024), 128)
    pool3 = dy.max_dim(l3, d = 1)

    my_norm = dy.l2_norm(pool3) + 1e-6
    output = dy.cdiv(pool3,my_norm)
    output = dy.reshape(output, (self.num_filters[2],), batch_size = batch_size)

    return ExpressionSequence(expr_tensor=output)


# This is an image encoder that takes in features and does a linear transform from the following paper
#  http://papers.nips.cc/paper/6186-unsupervised-learning-of-spoken-language-with-visual-context.pdf
class HarwathImageTransducer(SeqTransducer, Serializable):
  """
    Inputs are first put through 2 CNN layers, each with stride (2,2), so dimensionality
    is reduced by 4 in both directions.
    Then, we add a configurable number of bidirectional RNN layers on top.
    """
  yaml_tag = '!HarwathImageTransducer'

  @serializable_init
  def __init__(self, in_height, out_height):
    """
    Args:
      num_layers: depth of the RNN
      input_dim: size of the inputs
      hidden_dim: size of the outputs (and intermediate RNN layer representations)
      """

    model = ParamManager.my_params(self)
    self.in_height = in_height
    self.out_height = out_height

    normalInit=dy.NormalInitializer(0, 0.1)
    self.pW = model.add_parameters(dim = (self.out_height, self.in_height), init=normalInit)
    self.pb = model.add_parameters(dim = self.out_height)

  def transduce(self, src: ExpressionSequence) -> ExpressionSequence:
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

