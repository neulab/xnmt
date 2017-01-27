from vocab import *

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
  def __init__(self):
    self.vocab = Vocab()

  def read_file(self, filename):
    sentences = []
    with open(filename) as f:
      for line in f:
        words = line.decode('utf-8').strip().split()
        sentence = [self.vocab.convert(word) for word in words]
        sentence.append(self.vocab.convert('</s>'))
        sentences.append(sentence)
    return sentences
