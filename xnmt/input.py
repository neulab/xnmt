from vocab import *
import pickle

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
  def __init__(self, load_vocab_path=None):
    if not load_vocab_path is None:
      self.load_vocab(load_vocab_path)
    else:
      self.vocab = Vocab()

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

  def save_vocab(self, file_path):
    with open(file_path + ".vocab", 'wb') as handle:
      pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def load_vocab(self, file_path):
    with open(file_path + ".vocab", 'rb') as handle:
      self.vocab = pickle.load(handle)