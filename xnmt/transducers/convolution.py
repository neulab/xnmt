from typing import List
import numbers

import xnmt, xnmt.tensor_tools as tt
from xnmt import expression_seqs, param_collections
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init

if xnmt.backend_dynet:
  import dynet as dy

if xnmt.backend_torch:
  import torch.nn as nn

@xnmt.require_dynet
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
      output_dim: size of the outputs
      internal_dim: size of hidden dimension, internal dimension
      non_linearity: Non linearity to apply between layers
      """

    my_params = param_collections.ParamManager.my_params(self)
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

    self.pConv1 = my_params.add_parameters(dim = (self.input_dim,self.window_receptor,1,self.internal_dim),init=normalInit)
    self.pBias1 = my_params.add_parameters(dim = (self.internal_dim,))
    self.builder_layers = []
    for _ in range(num_layers):
        conv = my_params.add_parameters(dim = (self.internal_dim,1,1,self.internal_dim),init=normalInit)
        bias = my_params.add_parameters(dim = (self.internal_dim,))
        self.builder_layers.append((conv,bias))

    self.last_conv = my_params.add_parameters(dim = (self.internal_dim,1,1,self.output_dim),init=normalInit)
    self.last_bias = my_params.add_parameters(dim = (self.output_dim,))

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

@xnmt.require_torch
class MaxPoolCNNLayer(transducers.SeqTransducer, Serializable):
  """
  One layer of CNN + (potentially strided) max-pooling.


  """
  yaml_tag = "!MaxPoolCNNLayer"
  @serializable_init
  def __init__(self,
               in_channels: numbers.Integral, # 1 / 128
               out_channels: numbers.Integral, # 128 / 128
               kernel_h: numbers.Integral = 1, # 9 / 9
               kernel_w: numbers.Integral = 1, # 21 / 1
               pad_cnn_h: bool = False,
               pad_cnn_w: bool = False,
               pool_h: numbers.Integral = 1,   # 2
               pool_w: numbers.Integral = 1,   # 1
               pad_pool_h: bool = False,
               pad_pool_w: bool = False,
               stride_h: numbers.Integral = 1, # 2
               stride_w: numbers.Integral = 1, # 1
               activation: str = 'selu'):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.activation = activation
    my_params = param_collections.ParamManager.my_params(self)
    self.cnn_layer = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(kernel_h, kernel_w),
                               padding=(kernel_h // 2 if pad_cnn_h else 0,
                                        kernel_w // 2 if pad_cnn_w else 0))
    self.pooling_layer = nn.MaxPool2d(kernel_size=(pool_h, pool_w),
                                      stride=(stride_h, stride_w),
                                      padding=(pool_h // 2 if pad_pool_h else 0,
                                               pool_w // 2 if pad_pool_w else 0))
    my_params.append(self.cnn_layer)
    my_params.append(self.pooling_layer)
    self.activation_fct = tt.activation_by_name(activation)

  def transduce(self, x: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    expr = x.as_transposed_tensor()
    batch_size, hidden_dim, seq_len = expr.size()
    expr = expr.view((batch_size, self.in_channels, hidden_dim//self.in_channels, seq_len))
    expr = self.cnn_layer(expr)
    expr = self.pooling_layer(expr)
    expr = self.activation_fct(expr)
    batch_size, out_chn, out_h, seq_len = expr.size()
    expr = expr.view((batch_size, out_chn * out_h, seq_len))
    output_seq = expression_seqs.ExpressionSequence(expr_transposed_tensor = expr, mask = x.mask)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states
