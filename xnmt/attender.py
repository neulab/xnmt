import dynet as dy
from batcher import *


class Attender:
  '''
  A template class for functions implementing attention.
  '''

  '''
  Implement things.
  '''

  def start_sentence(self, sentence):
    raise NotImplementedError('start_sentence must be implemented for Attender subclasses')

  def calc_attention(self, state):
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')


class StandardAttender(Attender):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  def __init__(self, input_dim, state_dim, hidden_dim, model):
    self.pW = model.add_parameters((hidden_dim, input_dim))
    self.pV = model.add_parameters((hidden_dim, state_dim))
    self.pb = model.add_parameters(hidden_dim)
    self.pU = model.add_parameters((1, hidden_dim))
    self.curr_sentence = None
    self.serialize_params = [input_dim, state_dim, hidden_dim, model]

  def start_sentence(self, sentence):
    self.curr_sentence = sentence
    I = dy.concatenate_cols(self.curr_sentence)
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    self.WI = dy.affine_transform([b, W, I])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)

    return dy.softmax(scores)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = dy.concatenate_cols(self.curr_sentence)
    return I * attention
