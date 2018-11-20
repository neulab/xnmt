import dynet as dy
from typing import List
import numbers

from xnmt import expression_seqs, param_collections
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init

class ConvConnectedSeqTransducer(transducers.SeqTransducer, Serializable):
  yaml_tag = '!ConvConnectedSeqTransducer'
  """
    Input goes through through a first convolution in time and space, no stride,
    dimension is not reduced, then CNN layer for each frame several times
    Embedding sequence has same length as Input sequence
    """

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral,
               window_receptor: numbers.Integral,
               output_dim: numbers.Integral,
               num_layers: numbers.Integral,
               internal_dim: numbers.Integral,
               non_linearity: str = 'linear') -> None:
    """
    Args:
      num_layers: num layers after first receptor conv
      input_dim: size of the inputs
      window_receptor: window for the receptor
      ouput_dim: size of the outputs
      internal_dim: size of hidden dimension, internal dimension
      non_linearity: Non linearity to apply between layers
      """

    model = param_collections.ParamManager.my_params(self)
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

    normalInit=dy.NormalInitializer(0, 0.1)

    self.pConv1 = model.add_parameters(dim = (self.input_dim,self.window_receptor,1,self.internal_dim),init=normalInit)
    self.pBias1 = model.add_parameters(dim = (self.internal_dim,))
    self.builder_layers = []
    for _ in range(num_layers):
        conv = model.add_parameters(dim = (self.internal_dim,1,1,self.internal_dim),init=normalInit)
        bias = model.add_parameters(dim = (self.internal_dim,))
        self.builder_layers.append((conv,bias))

    self.last_conv = model.add_parameters(dim = (self.internal_dim,1,1,self.output_dim),init=normalInit)
    self.last_bias = model.add_parameters(dim = (self.output_dim,))

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, embed_sent: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    src = embed_sent.as_tensor()

    sent_len = src.dim()[0][1]
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
    output = dy.reshape(output, (sent_len,self.output_dim),batch_size=batch_size)
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq




