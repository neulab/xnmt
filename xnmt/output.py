from vocab import *
import pickle

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
  with one sentence per line.
  '''

  def load_vocab(self, file_path):
    with open(file_path + ".vocab", 'rb') as handle:
      self.vocab = pickle.load(handle)

  def process(self, input):
    self.token_strings = []
    for token_list in input:
      self.token_string = []
      for token in token_list:
        self.token_string.append(self.vocab[token])
    self.token_strings.append(self.to_string())
    return self.token_strings

  def to_string(self):
    output_str = ""
    for token in self.token_string:
      output_str = output_str + token + " "
    return output_str

