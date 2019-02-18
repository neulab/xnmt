class Decoder(object):
  """
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  """
  def initial_state(self, enc_final_states, ss_expr):
    raise NotImplementedError('must be implemented by subclasses')
  def add_input(self, dec_state, trg_word):
    raise NotImplementedError('must be implemented by subclasses')
  def calc_loss(self, dec_state, ref_action):
    raise NotImplementedError('must be implemented by subclasses')
  def best_k(self, dec_state, k, normalize_scores=False):
    raise NotImplementedError('must be implemented by subclasses')
  def sample(self, dec_state, n, temperature=1.0):
    raise NotImplementedError('must be implemented by subclasses')
  def init_sent(self, sent_enc):
    pass
  def eog_symbol(self):
    raise NotImplementedError('must be implemented by subclasses')
  def finish_generating(self, dec_output, dec_state):
    raise NotImplementedError('must be implemented by subclasses')


class DecoderState(object):
  """A state that holds whatever information is required for the decoder.
     Child classes must implement the as_vector() method, which will be
     used by e.g. the attention mechanism"""

  def as_vector(self):
    raise NotImplementedError('must be implemented by subclass')