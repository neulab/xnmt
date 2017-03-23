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

  def calc_attention(self, state):
    # TODO: This can be optimized by pre-computing WI in start_sentence
    W = dy.parameter(self.pW)
    V = dy.parameter(self.pV)
    b = dy.parameter(self.pb)
    U = dy.parameter(self.pU)

    I = dy.concatenate_cols(self.curr_sentence)
    WI = W * I
    Vsb = dy.affine_transform([b, V, state])
    Vsb_n = dy.concatenate_cols([Vsb] * len(self.curr_sentence))
    h = dy.tanh(WI + Vsb_n)
    scores = dy.transpose(U * h)

    return dy.softmax(scores)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = dy.concatenate_cols(self.curr_sentence)
    return I * attention
