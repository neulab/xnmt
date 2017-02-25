import dynet as dy
from mlp import MLP
import inspect

class Decoder:
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''
  
  '''
  Document me
  '''
  def calc_loss(self, x):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')

class MlpSoftmaxDecoder(Decoder):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.
  def __init__(self, layers, input_dim, lstm_dim, mlp_hidden_dim, embedder, model):
    self.embedder = embedder
    self.input_dim = input_dim
    self.fwd_lstm = dy.LSTMBuilder(layers, embedder.emb_dim, lstm_dim, model)
    self.mlp = MLP(input_dim + lstm_dim, mlp_hidden_dim, embedder.vocab_size, model)
    self.state = None

  def initialize(self):
    self.state = self.fwd_lstm.initial_state()

  def add_input(self, target_word):
    self.state = self.state.add_inputs([self.embedder.embed(target_word)])[-1]

  def calc_loss(self, context, ref_action):
    mlp_input = dy.concatenate([context, self.state.output()])
    scores = self.mlp(mlp_input)
    return dy.pickneglogsoftmax(scores, ref_action)
