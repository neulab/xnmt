import dynet as dy
from batcher import *
from serializer import *
import model_globals


class Attender:
  '''
  A template class for functions implementing attention.
  '''

  def __init__(self, input_dim):
    """
    :param input_dim: every attender needs an input_dim
    """
    pass

  def start_sent(self, sent):
    raise NotImplementedError('start_sent must be implemented for Attender subclasses')

  def calc_attention(self, state):
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')


class StandardAttender(Attender, Serializable):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  yaml_tag = u'!StandardAttender'

  def __init__(self, input_dim, state_dim, hidden_dim):
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = model_globals.get("dynet_param_collection")
    self.pW = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb = param_collection.add_parameters(hidden_dim)
    self.pU = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent = None

  def start_sent(self, sent):
    self.curr_sent = sent
    I = dy.concatenate_cols(self.curr_sent)
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    self.WI = dy.affine_transform([b, W, I])
    if len(self.curr_sent)==1:
      self.WI = dy.reshape(self.WI, (self.WI.dim()[0][0],1), batch_size=self.WI.dim()[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)

    return dy.softmax(scores)

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = dy.concatenate_cols(self.curr_sent)
    return I * attention

