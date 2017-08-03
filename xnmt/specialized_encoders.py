import dynet as dy
from attender import *
from encoder import *
import numpy as np
from expression_sequence import *

# This is a file for specialized encoders that implement a particular model
# Ideally, these will eventually be refactored to use standard components and the ModularEncoder framework,
#  (for more flexibility), but for ease of implementation it is no problem to perform an initial implementation here.

# This is a CNN-based encoder that was used in the following paper:
#  http://papers.nips.cc/paper/6186-unsupervised-learning-of-spoken-language-with-visual-context.pdf
class TilburgSpeechEncoder(Encoder, Serializable):
  yaml_tag = u'!TilburgSpeechEncoder'
  def __init__(self, filter_height, filter_width, channels, num_filters, stride, rhn_num_hidden_layers, rhn_dim, rhn_microsteps, attention_dim, residual= False):
    """
    :param etc.

    """
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
    
    normalInit=dy.NormalInitializer(0, 0.1)
    model = model_globals.dynet_param_collection.param_col
    # Convolutional layer
    self.filter_conv = model.add_parameters(dim=(self.filter_height, self.filter_width, self.channels, self.num_filters), init=normalInit)
  
    # Recurrent highway layer
    self.recurH = []
    self.recurT = []
    self.linearH = []
    self.linearT = []
    
    for l in range(rhn_num_hidden_layers):
      recurH_layer = []
      recurT_layer = []
      if l == 0:
        self.linearH.append(FullyConnectedEncoder(num_filters, rhn_dim, 'linear', with_bias=True))
        self.linearT.append(FullyConnectedEncoder(num_filters, rhn_dim, 'linear', with_bias=True))
      else:
        self.linearH.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'linear', with_bias=True))
        self.linearT.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'linear', with_bias=True))
      for m in range(self.rhn_microsteps):
        if m  == 0:
          # Special case for the input layer
          recurH_layer.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'linear', with_bias=False))
          recurT_layer.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'linear', with_bias=False))
        else:
          recurH_layer.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'sigmoid'))
          recurT_layer.append(FullyConnectedEncoder(rhn_dim, rhn_dim, 'sigmoid'))
      self.recurH.append(recurH_layer)
      self.recurT.append(recurT_layer)
    
    # Attention layer  
    self.attender = StandardAttender(self.rhn_dim, self.rhn_dim, attention_dim)
    
  def transduce(self, src):
    src = src.as_tensor()
    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    src_channels = 1
    batch_size = src.dim()[1]
    src = dy.reshape(src, (src_height, src_width, src_channels), batch_size=batch_size)
       
    ''' Check the activation function '''
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filter_conv), stride = [self.stride, self.stride], is_valid = True))
    timestep = l1.npvalue().shape[1]
    rhn_in = dy.transpose(dy.reshape(l1, (timestep, l1.npvalue().shape[2]), batch_size = batch_size))
   
    for l in range(self.rhn_num_hidden_layers):
      # initialize a random vector for the first state vector 
      print('layer',l)
      rhn_out = []
      prev_state = dy.inputTensor(np.random.normal(size=(self.rhn_dim,)))
      for t in range(timestep): 
        for m in range(self.rhn_microsteps):
          # Recurrent step
          if m == 0:
            H = dy.tanh(self.linearH[l].transduce(ExpressionSequence(expr_tensor = dy.pick(rhn_in, t, 1))).as_tensor() + self.recurH[l][m].transduce(ExpressionSequence(expr_tensor = prev_state)).as_tensor())
            T = dy.logistic(self.linearT[l].transduce(ExpressionSequence(expr_tensor = dy.pick(rhn_in, t, 1))).as_tensor() + self.recurT[l][m].transduce(ExpressionSequence(expr_tensor = prev_state)).as_tensor()) 
         
            #print('\n===> Check the value  of the gates across batch: ', (self.linearT[l].transduce(ExpressionSequence(expr_tensor = dy.pick(rhn_in, t, 1))).as_tensor()).npvalue()[0]) 
          else: 
            H = self.recurH[l][m].transduce(ExpressionSequence(expr_tensor = prev_state))
            T = self.recurT[l][m].transduce(ExpressionSequence(expr_tensor = prev_state))
            H = H.as_tensor()
            T = T.as_tensor()
          
          prev_state = dy.cmult(dy.inputTensor(np.ones((self.rhn_dim, batch_size)), batched = True) - T, prev_state) + dy.cmult(T, H)
         #print('\n===> Check the dimention of the state, expected ((1024,), batch_size) : ', prev_state.npvalue()[:, 0], prev_state.npvalue().shape[0])
        rhn_out.append(prev_state)
      rhn_in = dy.reshape(dy.concatenate(rhn_out), (self.rhn_dim, timestep), batch_size = batch_size) 
      #print('\n===> Check the dimention of rhn_in update, expected ((1024, 510), batch_size)', rhn_in.npvalue().shape[0])
      #print('\n===> Check the gate value: ', H.npvalue()[0], T.npvalue()[0])
      #print('\n===> Check whether nan is in rhn_in, expected False:', rhn_in.npvalue(), np.any(rhn_in.npvalue() == float('nan')))
    # Compute the attention-weighted average of the activations
    #attn_out = dy.max_dim(rhn_in, d = 1)
    self.attender.start_sent(ExpressionSequence(expr_tensor=rhn_in))
    attn_out = self.attender.calc_context(dy.inputTensor(np.zeros((self.rhn_dim))))
    print('forward finished')
    return ExpressionSequence(expr_tensor = attn_out)

  def initial_state(self):
    return PseudoState(self)

 
      
class HarwathSpeechEncoder(Encoder, Serializable):
  yaml_tag = u'!HarwathSpeechEncoder'
  def __init__(self, filter_height, filter_width, channels, num_filters, stride):
    """
    :param num_layers: depth of the RNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
    #model = model_globals.dynet_param_collection.param_col
    self.filter_height = filter_height
    self.filter_width = filter_width
    self.channels = channels
    self.num_filters = num_filters
    self.stride = stride # (2,2)
    # hidden states is a dictionary whose keys 'l1', 'l2', 'output' correspond to hidder layer 1, 2 etc.  and the final should be the output of the encoder.
    # its values are the dynet expressions of those hidden state.
    self.hidden_states = {}

    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters1 = model.add_parameters(dim=(self.filter_height[0], self.filter_width[0], self.channels[0], self.num_filters[0]),
                                         init=normalInit)
    self.filters2 = model.add_parameters(dim=(self.filter_height[1], self.filter_width[1], self.channels[1], self.num_filters[1]),
                                         init=normalInit)
    self.filters3 = model.add_parameters(dim=(self.filter_height[2], self.filter_width[2], self.channels[2], self.num_filters[2]),
                                         init=normalInit)

  def transduce(self, src):
    src = src.as_tensor()

    src_height = src.dim()[0][0]
    src_width = src.dim()[0][1]
    src_channels = 1
    batch_size = src.dim()[1]

    src = dy.reshape(src, (src_height, src_width, src_channels), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers
    l1 = dy.rectify(dy.conv2d(src, dy.parameter(self.filters1), stride = [self.stride[0], self.stride[0]], is_valid = True))
    pool1 = dy.maxpooling2d(l1, (1, 4), (1,2), is_valid = True)
    self.hidden_states['l1'] = pool1

    l2 = dy.rectify(dy.conv2d(pool1, dy.parameter(self.filters2), stride = [self.stride[1], self.stride[1]], is_valid = True))
    pool2 = dy.maxpooling2d(l2, (1, 4), (1,2), is_valid = True)
    self.hidden_states['l2'] = pool2

    l3 = dy.rectify(dy.conv2d(pool2, dy.parameter(self.filters3), stride = [self.stride[2], self.stride[2]], is_valid = True))
    pool3 = dy.max_dim(l3, d = 1)
    # print(pool3.dim())
    my_norm = dy.l2_norm(pool3) + 1e-6
    output = dy.cdiv(pool3,my_norm)
    output = dy.reshape(output, (self.num_filters[2],), batch_size = batch_size)
    self.hidden_states['output'] = output
    # print("my dim: ", output.dim())

    return ExpressionSequence(expr_tensor=output)
  def get_hidden_states(self):
    return self.hidden_states
  
  def initial_state(self):
    return PseudoState(self)


# This is an image encoder that takes in features and does a linear transform from the following paper
#  http://papers.nips.cc/paper/6186-unsupervised-learning-of-spoken-language-with-visual-context.pdf
class HarwathImageEncoder(Encoder, Serializable):
  yaml_tag = u'!HarwathImageEncoder'
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

    model = model_globals.dynet_param_collection.param_col
    self.in_height = in_height
    self.out_height = out_height

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

    src = dy.reshape(src, (src_height, src_width), batch_size=batch_size) # ((276, 80, 3), 1)
    # convolution and pooling layers
    l1 = (W*src)+b
    output = dy.cdiv(l1,dy.sqrt(dy.squared_norm(l1)))
    return ExpressionSequence(expr_tensor=output)

  def initial_state(self):
    return PseudoState(self)

if __name__ == '__main__':
  model = dy.ParameterCollection()
  src = dy.inputTensor([np.random.normal(size=(37, 1024)), np.random.normal(size=(37, 1024))])
  encoder = TilburgSpeechEncoder(37, 6, 1, 64, 2, 2, 1024, 2, 128)
  out = encoder.transduce(src)
  print(out.dim())
