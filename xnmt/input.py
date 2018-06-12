from typing import Any, Sequence

import numpy as np

from xnmt import vocab

class Input(object):
  """
  A template class to represent a single input of any type.
  """
  def __len__(self):
    """
    Return length of input, included padded tokens.

    Returns: length
    """
    raise NotImplementedError("__len__() must be implemented by Input subclasses")

  def len_unpadded(self):
    """
    Return length of input prior to applying any padding.

    Returns: unpadded length

    """

  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")

  def get_padded_sent(self, token: Any, pad_len: int) -> 'Input':
    """
    Return padded version of the sentence.
    
    Args:
      token: padding token
      pad_len: number of tokens to append
    Returns:
      padded sentence
    """
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sentence, represented as a list of tokens
  
  Args:
    words (List[int]): list of integer word ids
    vocab (Vocab):
  """

  def __init__(self, words: Sequence[int], vocab: vocab.Vocab = None):
    self.words = words
    self.vocab = vocab

  def __len__(self):
    return len(self.words)

  def len_unpadded(self):
    return sum(x!=vocab.Vocab.ES for x in self.words)

  def __getitem__(self, key):
    return self.words[key]

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.
    
    Args:
      token (int): padding token
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.SimpleSentenceInput: padded sentence
    """
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([token] * pad_len)
    return self.__class__(new_words, self.vocab)

  def __str__(self):
    return " ".join(map(str, self.words))

class AnnotatedSentenceInput(SimpleSentenceInput):
  def __init__(self, words, vocab=None):
    super(AnnotatedSentenceInput, self).__init__(words, vocab)
    self.annotation = {}

  def annotate(self, key, value):
    self.annotation[key] = value

  def get_padded_sent(self, token, pad_len):
    sent = super(AnnotatedSentenceInput, self).get_padded_sent(token, pad_len)
    sent.annotation = self.annotation
    return sent

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens.
  
  Args:
    nparr: numpy array
  """
  def __init__(self, nparr: np.ndarray, padded_len: int = 0):
    self.nparr = nparr
    self.padded_len = padded_len

  def __len__(self):
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def len_unpadded(self):
    return len(self) - self.padded_len

  def __getitem__(self, key):
    return self.nparr.__getitem__(key)

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.
    
    Args:
      token: None (replicate last frame) or 0 (pad zeros)
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.ArrayInput: padded sentence
    """
    if pad_len == 0:
      return self
    if token is None:
      new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:,-1], (self.nparr.shape[0], 1)), (self.nparr.shape[0], pad_len)), axis=1)
    elif token == 0:
      new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
    else:
      raise NotImplementedError(f"currently only support 'None' or '0' as, but got '{token}'")
    return ArrayInput(new_nparr, padded_len=self.padded_len + pad_len)

  def get_array(self):
    return self.nparr