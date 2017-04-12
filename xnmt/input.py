from vocab import *

import numpy as np

class Input:
  '''
  A template class to represent all inputs.
  '''
  pass

class InputReader:
  @staticmethod
  def create_input_reader(input_type, vocab=None):
    if input_type == "word":
      return PlainTextReader(vocab)
    elif input_type == "feat-vec":
      return FeatVecReader()
    else:
      raise RuntimeError("Unkonwn input type {}".format(input_type))


class PlainTextReader(InputReader):
  '''
  Handles the typical case of reading plain text files,
  with one sentence per line.
  '''
  def __init__(self, vocab=None):
    if vocab is None:
      self.vocab = Vocab()
    else:
      self.vocab = vocab

  def read_file(self, filename):
    sentences = []
    with open(filename) as f:
      for line in f:
        words = line.strip().split()
        sentence = [self.vocab.convert(word) for word in words]
        sentence.append(self.vocab.convert('</s>'))
        sentences.append(sentence)
    return sentences

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk('UNK')

class FeatVecReader(InputReader):
  '''
  Handles the case where sentences are sequences of feature vectors.
  We assumine one sentence per line, words are separated by semicolons, vector entries by 
  whitespace. E.g.:
  2.3 4.2;5.1 3
  2.3 4.2;1 -1;5.1 3
  
  TODO: should probably move to a binary format, as these files can get quite large.
  '''
  def __init__(self):
    self.vocab = Vocab()

  def read_file(self, filename):
    sentences = []
    with open(filename) as f:
      for line in f:
        words = line.strip().split(";")
        sentence = [np.asarray([float(x) for x in word.split()]) for word in words]
        sentences.append(sentence)
    return sentences

  def freeze(self):
    pass
