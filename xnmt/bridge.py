import dynet as dy

import xnmt.linear
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path

class Bridge(object):
  """
  Responsible for initializing the decoder LSTM, based on the final encoder state
  """
  def decoder_init(self, dec_layers, dec_dim, enc_final_states):
    raise NotImplementedError("decoder_init() must be implemented by Bridge subclasses")

class NoBridge(Bridge, Serializable):
  """
  This bridge initializes the decoder with zero vectors, disregarding the encoder final states.
  """
  yaml_tag = u'!NoBridge'
  def __init__(self, dec_layers, dec_dim = None, xnmt_global=Ref(Path("xnmt_global"))):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or xnmt_global.default_layer_dim
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
  """
  yaml_tag = u'!CopyBridge'
  def __init__(self, dec_layers, dec_dim = None, xnmt_global=Ref(Path("xnmt_global"))):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or xnmt_global.default_layer_dim
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
  Requires that:
  - num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)
  """
  yaml_tag = u'!LinearBridge'
  def __init__(self, dec_layers, enc_dim = None, dec_dim = None, xnmt_global=Ref(Path("xnmt_global"))):
    param_col = xnmt_global.dynet_param_collection.param_col
    self.dec_layers = dec_layers
    self.enc_dim = enc_dim or xnmt_global.default_layer_dim
    self.dec_dim = dec_dim or xnmt_global.default_layer_dim
    self.projector = xnmt.linear.Linear(input_dim  = enc_dim,
                                           output_dim = dec_dim,
                                           model = param_col)
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("LinearBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.enc_dim:
      raise RuntimeError("LinearBridge requires enc_dim == %s, but got %s" % (self.enc_dim, enc_final_states[0].main_expr().dim()[0][0]))
    decoder_init = [self.projector(enc_state.main_expr()) for enc_state in enc_final_states[-self.dec_layers:]]
    return decoder_init + [dy.tanh(dec) for dec in decoder_init]
