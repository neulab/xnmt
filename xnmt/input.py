from typing import Any, Sequence
import warnings

import numpy as np
import functools
import copy
from xnmt.vocab import Vocab

from xnmt import vocab


class Input(object):
  """
  A template class to represent a single input of any type.
  """
  def __len__(self) -> int:
    warnings.warn("use of Input.__len__() is discouraged, use Input.sent_len() instead.", DeprecationWarning)
    return self.sent_len()
  
  def sent_len(self) -> int:
    """
    Return length of input, included padded tokens.

    Returns: length
    """
    raise NotImplementedError("must be implemented by Input subclasses")

  def len_unpadded(self) -> int:
    """
    Return length of input prior to applying any padding.

    Returns: unpadded length

    """

  def __getitem__(self) -> Any:
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

  def get_truncated_sent(self, trunc_len: int) -> 'Input':
    """
    Return right-truncated version of the sentence.

    Args:
      trunc_len: number of tokens to truncate
    Returns:
      truncated sentence
    """
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")


class IntInput(Input):
  def __init__(self, value: int) -> None:
    self.value = value
  def sent_len(self) -> int:
    return 1
  def len_unpadded(self) -> int:
    return 1
  def __getitem__(self, item) -> int:
    if item != 0: raise IndexError
    return self.value
  def get_padded_sent(self, token: Any, pad_len: int) -> 'IntInput':
    if pad_len != 0:
      raise ValueError("Can't pad IntInput")
    return self
  def get_truncated_sent(self, trunc_len: int) -> 'IntInput':
    if trunc_len != 0:
      raise ValueError("Can't truncate IntInput")
    return self


class CompoundInput(Input):
  def __init__(self, inputs: Sequence[Input]) -> None:
    self.inputs = inputs
  def sent_len(self) -> int:
    return sum(inp.sent_len() for inp in self.inputs)
  def len_unpadded(self) -> int:
    return sum(inp.len_unpadded() for inp in self.inputs)
  def __getitem__(self, key):
    raise ValueError("not supported with CompoundInput, must be called on one of the sub-inputs instead.")
  def get_padded_sent(self, token, pad_len):
    raise ValueError("not supported with CompoundInput, must be called on one of the sub-inputs instead.")
  def get_truncated_sent(self, trunc_len):
    raise ValueError("not supported with CompoundInput, must be called on one of the sub-inputs instead.")

class SimpleSentenceInput(Input):
  """
  A simple sentence, represented as a list of tokens

  Args:
    words: list of integer word ids
  """

  def __init__(self, words: Sequence[int]):
    self.words = words

  def __repr__(self):
    return '{}'.format(self.words)

  def sent_len(self):
    return len(self.words)
  
  @functools.lru_cache(maxsize=1)
  def len_unpadded(self):
    return sum(x != vocab.Vocab.ES for x in self.words)

  def __getitem__(self, key):
    ret = self.words[key]
    if isinstance(ret, list): # support for slicing
      return SimpleSentenceInput(ret)
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
    # Copy is used to copy all possible annotations
    new_sent = copy.deepcopy(self)
    new_sent.words.extend([token] * pad_len)
    return new_sent

  def get_truncated_sent(self, trunc_len: int) -> 'Input':
    if trunc_len == 0:
      return self
    new_sent = copy.deepcopy(self)
    new_sent.words = self.words[:-trunc_len]
    return new_sent


  def __str__(self):
    return " ".join(map(str, self.words))

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens.

  Args:
    nparr: numpy array
  """

  def __init__(self, nparr: np.ndarray, padded_len: int = 0):
    self.nparr = nparr
    self.padded_len = padded_len

  def sent_len(self):
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
      new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:, -1], (self.nparr.shape[0], 1)),
                                                        (self.nparr.shape[0], pad_len)), axis=1)
    elif token == 0:
      new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
    else:
      raise NotImplementedError(f"currently only support 'None' or '0' as, but got '{token}'")
    return ArrayInput(new_nparr, padded_len=self.padded_len + pad_len)

  def get_array(self):
    return self.nparr

