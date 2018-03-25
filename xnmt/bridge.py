import dynet as dy

import xnmt.linear
from xnmt.serialize.serializable import Serializable, Ref, Path
from xnmt.serialize.serializer import serializable_init

class Bridge(object):
  """
  Responsible for initializing the decoder LSTM, based on the final encoder state
  """
  def decoder_init(self, dec_layers, dec_dim, enc_final_states):
    """
    Args:
      dec_layers (int): number of decoder layers
      dec_dim (int): dimension of decoder layers
      enc_final_states (List[FinalTransducerState]): list of final states for each encoder layer
    Returns:
      list of dy.Expression: list of initial hidden and cell expressions for each layer. List indices 0..n-1 hold hidden states, n..2n-1 hold cell states.
    """
    raise NotImplementedError("decoder_init() must be implemented by Bridge subclasses")

class NoBridge(Bridge, Serializable):
  """
  This bridge initializes the decoder with zero vectors, disregarding the encoder final states.

  Args:
    dec_layers (int): number of decoder layers to initialize
    dec_dim (int): hidden dimension of decoder states; if None, use exp_global.default_layer_dim
    exp_global (ExpGlobal): ExpGlobal object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
  """
  yaml_tag = '!NoBridge'

  @serializable_init
  def __init__(self, dec_layers = 1, dec_dim = None, exp_global=Ref(Path("exp_global"))):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or exp_global.default_layer_dim
  def decoder_init(self, enc_final_states):
    batch_size = enc_final_states[0].main_expr().dim()[1]
    z = dy.zeros(self.dec_dim, batch_size)
    return [z] * (self.dec_layers * 2)

class CopyBridge(Bridge, Serializable):
  """
  This bridge copies final states from the encoder to the decoder initial states.
  Requires that:
  - encoder / decoder dimensions match for every layer
  - num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)
  
  Args:
    dec_layers (int): number of decoder layers to initialize
    dec_dim (int): hidden dimension of decoder states
  """
  yaml_tag = '!CopyBridge'

  @serializable_init
  def __init__(self, dec_layers = 1, dec_dim = Ref(Path("exp_global.default_layer_dim"))):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("CopyBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.dec_dim:
      raise RuntimeError("CopyBridge requires enc_dim == dec_dim, but got %s and %s" % (enc_final_states[0].main_expr().dim()[0][0], self.dec_dim))
    return [enc_state.cell_expr() for enc_state in enc_final_states[-self.dec_layers:]] \
         + [enc_state.main_expr() for enc_state in enc_final_states[-self.dec_layers:]]

class LinearBridge(Bridge, Serializable):
  """
  This bridge does a linear transform of final states from the encoder to the decoder initial states.
  Requires that  num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)
  
  Args:
    dec_layers (int): number of decoder layers to initialize
    enc_dim (int): hidden dimension of encoder states; if None, use exp_global.default_layer_dim
    dec_dim (int): hidden dimension of decoder states; if None, use exp_global.default_layer_dim
    exp_global (ExpGlobal): ExpGlobal object to acquire DyNet params and global settings. By default, references the experiment's top level exp_global object.
    param_init (ParamInitializer): how to initialize weight matrices; if None, use ``exp_global.param_init``
    bias_init (ParamInitializer): how to initialize bias vectors; if None, use ``exp_global.bias_init``
  """
  yaml_tag = '!LinearBridge'

  @serializable_init
  def __init__(self, dec_layers = 1, enc_dim = None, dec_dim = None, exp_global=Ref(Path("exp_global")), param_init=None, bias_init=None):
    param_col = exp_global.dynet_param_collection.param_col
    self.dec_layers = dec_layers
    self.enc_dim = enc_dim or exp_global.default_layer_dim
    self.dec_dim = dec_dim or exp_global.default_layer_dim
    self.projector = xnmt.linear.Linear(input_dim  = enc_dim,
                                           output_dim = dec_dim,
                                           model = param_col,
                                           param_init = param_init or exp_global.param_init,
                                           bias_init = bias_init or exp_global.bias_init)
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("LinearBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.enc_dim:
      raise RuntimeError("LinearBridge requires enc_dim == %s, but got %s" % (self.enc_dim, enc_final_states[0].main_expr().dim()[0][0]))
    decoder_init = [self.projector(enc_state.main_expr()) for enc_state in enc_final_states[-self.dec_layers:]]
    return decoder_init + [dy.tanh(dec) for dec in decoder_init]
