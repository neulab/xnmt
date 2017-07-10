import math
import dynet as dy
from residual import PseudoState
from embedder import ExpressionSequence

class speechBuilder(object):
  
  def __init__(self, filter_height, filter_width, channels, num_filters, model, stride):
    """
    :param num_layers: depth of the RNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
  
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
    # convolution and pooling layers    
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filters1), stride = self.stride[0], is_valid = True))
    pool1 = dy.maxpooling2d(l1, (1, 4), (1,2), is_valid = True)

    l2 = dy.rectify(dy.conv2d(pool1, dy.parameter(self.filters2), stride = self.stride[1], is_valid = True))
    pool2 = dy.maxpooling2d(l2, (1, 4), (1,2), is_valid = True)

    l3 = dy.rectify(dy.conv2d(pool2, dy.parameter(self.filters3), stride = self.stride[2], is_valid = True))

    pool3 = dy.kmax_pooling(l3, 1, d = 1)
    print(pool3.dim())
    output = dy.cdiv(pool3,dy.sqrt(dy.squared_norm(pool3)))
    
    return output

  def initial_state(self):
    return PseudoState(self)
