import dynet as dy
import serializer
import model_globals
import batcher
import model
import mlp
import linear

from decorators import recursive, recursive_assign

# Short Name
HierarchicalModel = model.HierarchicalModel
Serializable = serializer.Serializable
param_col = lambda: model_globals.dynet_param_collection.param_col

class Decoder(HierarchicalModel):
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''

  '''
  Document me
  '''

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')

class RnnDecoder(Decoder):
  @staticmethod
  def rnn_from_spec(spec, num_layers, input_dim, hidden_dim, model, residual_to_output):
    decoder_type = spec.lower()
    if decoder_type == "lstm":
      return dy.CompactVanillaLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    elif decoder_type == "residuallstm":
      return residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim,
                                         model, residual_to_output)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!MlpSoftmaxDecoder'

  def __init__(self, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=False,
               bridge=None):
    # Define dim
    lstm_dim       = model_globals.default_if_none(lstm_dim)
    mlp_hidden_dim = model_globals.default_if_none(mlp_hidden_dim)
    trg_embed_dim  = model_globals.default_if_none(trg_embed_dim)
    input_dim      = model_globals.default_if_none(input_dim)
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += lstm_dim
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col(),
                                              residual_to_output = residual_to_output)
    # MLP
    self.context_projector = linear.Linear(input_dim  = input_dim + lstm_dim,
                                           output_dim = mlp_hidden_dim,
                                           model = param_col())
    self.vocab_projector = linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = param_col())
    # Dropout
    self.dropout = dropout or model_globals.get("dropout")
    # Mutable state
    self.state = None
    self.h_t = None

  def shared_params(self):
    return [set(["layers", "bridge.dec_layers"]),
            set(["lstm_dim", "bridge.dec_dim"])]

  def initialize(self, enc_final_states):
    dec_state = self.fwd_lstm.initial_state()
    self.state = dec_state.set_s(self.bridge.decoder_init(enc_final_states))
    self.h_t = None

  def add_input(self, trg_embedding):
    inp = trg_embedding
    if self.input_feeding:
      if self.h_t is not None:
        # Append with the last state of the decoder
        inp = dy.concatenate([inp, self.h_t])
      else:
        # Append with zero
        zero = dy.zeros(self.lstm_dim, batch_size=inp.dim()[1])
        inp = dy.concatenate([inp, zero])
    # The next state of the decoder
    self.state = self.state.add_input(inp)

  def get_scores(self, context):
    self.h_t = self.context_projector(dy.concatenate([context, self.state.output()]))
    return self.vocab_projector(self.h_t)

  def calc_loss(self, context, ref_action):
    scores = self.get_scores(context)
    # single mode
    if not batcher.is_batched(ref_action):
      return dy.pickneglogsoftmax(scores, ref_action)
    # minibatch mode
    else:
      return dy.pickneglogsoftmax_batch(scores, ref_action)

  @recursive
  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)

class Bridge(Serializable):
  """
  Responsible for initializing the decoder LSTM, based on the final encoder state
  """
  def decoder_init(self, dec_layers, dec_dim, enc_final_states):
    raise NotImplementedError("decoder_init() must be implemented by Bridge subclasses")

class NoBridge(Bridge):
  yaml_tag = u'!NoBridge'
  def __init__(self, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or model_globals.get("default_layer_dim")
  def decoder_init(self, enc_final_states):
    batch_size = enc_final_states[0].main_expr().dim()[1]
    z = dy.zeros(self.dec_dim, batch_size)
    return [z] * (self.dec_layers * 2)

class CopyBridge(Bridge):
  yaml_tag = u'!CopyBridge'
  def __init__(self, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or model_globals.get("default_layer_dim")
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states): 
      raise RuntimeError("CopyBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.dec_dim:
      raise RuntimeError("CopyBridge requires enc_dim == dec_dim, but got %s and %s" % (enc_final_states[0].main_expr().dim()[0][0], self.dec_dim))
    return [enc_state.cell_expr() for enc_state in enc_final_states[-self.dec_layers:]] \
         + [enc_state.main_expr() for enc_state in enc_final_states[-self.dec_layers:]]
    