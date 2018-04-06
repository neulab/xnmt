import logging
logger = logging.getLogger('xnmt')

import h5py
import numpy as np
from xnmt.vocab import Vocab

class Input(object):
  """
  A template class to represent a single input of any type.
  """
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Input subclasses")

  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.

    Args:
      token: padding token
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.Input: padded sent
    """
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sent, represented as a list of tokens

  Args:
    words (List[int]): list of integer word ids
    Annotation (Dict): annotation for the sentence
  """
  def __init__(self, words, annotation=None):
    self.words = words
    self.annotation = annotation or {}
    self.original_length = words.index(Vocab.ES)+1

  def __len__(self):
    return len(self.words)

  def __getitem__(self, key):
    return self.words[key]

  def annotate(self, key, value):
    self.annotation[key] = value

  def get_annotation(self, key):
    return self.annotation[key]

  def has_annotation(self, key):
    return key in self.annotation

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
    return self.__class__(new_words, self.annotation)

  def __str__(self):
    return " ".join(map(str, self.words))

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens.

  Args:
    nparr: numpy array
  """
  def __init__(self, nparr):
    self.nparr = nparr

  def __len__(self):
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

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
    return ArrayInput(new_nparr)

  def get_array(self):
    return self.nparr

