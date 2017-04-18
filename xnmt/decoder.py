import dynet as dy
from mlp import MLP
import inspect
from batcher import *

class Decoder:
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''

  '''
  Document me
  '''

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')


class MlpSoftmaxDecoder(Decoder):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.
  def __init__(self, layers, input_dim, lstm_dim, mlp_hidden_dim, embedder, model, fwd_lstm_builder=dy.LSTMBuilder):
    self.embedder = embedder
    self.input_dim = input_dim
    self.fwd_lstm = fwd_lstm_builder(layers, embedder.emb_dim, lstm_dim, model)
    self.mlp = MLP(input_dim + lstm_dim, mlp_hidden_dim, embedder.vocab_size, model)
    self.state = None
    self.serialize_params = [layers, input_dim, lstm_dim, mlp_hidden_dim, embedder, model]

  def initialize(self):
    self.state = self.fwd_lstm.initial_state()

  def add_input(self, target_word):
    self.state = self.state.add_input(self.embedder.embed(target_word))

  def get_scores(self, context):
    mlp_input = dy.concatenate([context, self.state.output()])
    scores = self.mlp(mlp_input)
    return scores

  def calc_loss(self, context, ref_action):
    scores = self.get_scores(context)
    # single mode
    if not Batcher.is_batch(ref_action):
      return dy.pickneglogsoftmax(scores, ref_action)
    # minibatch mode
    else:
      return dy.pickneglogsoftmax_batch(scores, ref_action)
