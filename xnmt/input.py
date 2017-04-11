from vocab import *

import numpy as np

class Input:
  '''
  A template class to represent all inputs.
  '''
  pass

class InputReader:
  pass

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
