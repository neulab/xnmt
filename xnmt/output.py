from vocab import *

class Output:
  '''
  A template class to represent all output.
  '''
  def __init__(self, actions=None):
    ''' Initialize an output with actions. '''
    if actions is None:
      self.actions = []
    else:
      self.actions = actions

  def to_string(self):
    raise NotImplementedError('All outputs must implement to_string.')


class PlainTextOutput(Output):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''

  def load_vocab(self, vocab):
    self.vocab = vocab

  def process(self, inputs):
    self.token_strings = []
    filtered_tokens = set([Vocab.SS, Vocab.ES])
    for token_list in inputs:
      self.token_string = []
      for token in token_list:
        if token not in filtered_tokens:
          self.token_string.append(self.vocab[token])
    self.token_strings.append(self.to_string())
    return self.token_strings

  def to_string(self):
    return " ".join(self.token_string)

class JoinedCharTextOutput(PlainTextOutput):
  '''
  Assumes a single-character vocabulary and joins them to form words;
  per default, double underscores '__' are converted to spaces   
  '''
  def __init__(self, space_token='__'):
    self.space_token = space_token
  def to_string(self):
    return "".join(map(lambda s: ' ' if s==self.space_token else s, self.token_string))

class JoinedBPETextOutput(PlainTextOutput):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged   
  '''
  def __init__(self, merge_indicator='@'):
    self.merge_indicator_with_space = merge_indicator + " "
  def to_string(self):
    return " ".join(self.token_string).replace(self.merge_indicator_with_space, "")
