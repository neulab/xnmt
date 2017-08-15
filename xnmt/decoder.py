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
      return dy.VanillaLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    elif decoder_type == "residuallstm":
      return residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim,
                                         model, dy.VanillaLSTMBuilder,
                                         residual_to_output)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!MlpSoftmaxDecoder'

  def __init__(self, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=False):
    super(MlpSoftmaxDecoder, self).__init__()
    # Define dim
    lstm_dim       = model_globals.default_if_none(lstm_dim)
    mlp_hidden_dim = model_globals.default_if_none(mlp_hidden_dim)
    trg_embed_dim  = model_globals.default_if_none(trg_embed_dim)
    input_dim      = model_globals.default_if_none(input_dim)
    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = trg_embed_dim,
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
    # Input feeding
    self.input_feeding = input_feeding
    if input_feeding:
      self.feeding_projector = linear.Linear(input_dim = trg_embed_dim + lstm_dim,
                                             output_dim = trg_embed_dim,
                                             model = param_col())
    # Dropout
    self.dropout = dropout or model_globals.get("dropout")
    # Mutable state
    self.state = None
    self.h_t = None

  def initialize(self, last_encoding):
    state = self.fwd_lstm.initial_state()
    state = state.set_s(es=[last_encoding, dy.tanh(last_encoding)])
    self.state = state
    self.h_t = None

  def add_input(self, trg_embedding):
    inp = trg_embedding
    if self.input_feeding and self.h_t is not None:
      inp = self.feeding_projector(dy.concatenate([self.h_t, inp]))
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

