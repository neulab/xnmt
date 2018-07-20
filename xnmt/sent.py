from typing import List, Optional, Sequence
import functools
import copy

import numpy as np

from xnmt.vocab import Vocab
from xnmt.output import OutputProcessor

# TODO: add score: to each sentence, or a special class?

class Sentence(object):
  """
  A template class to represent a single data example of any type, used for both model input and output.

  Args:
    idx: running sentence number (unique among sentences loaded from the same file, but not across files)
  """

  def __init__(self, idx: int) -> None:
    self.idx = idx

  def sent_len(self) -> int:
    """
    Return length of input, included padded tokens.

    Returns: length
    """
    raise NotImplementedError("must be implemented by subclasses")

  def len_unpadded(self) -> int:
    """
    Return length of input prior to applying any padding.

    Returns: unpadded length
    """

  def create_padded_sent(self, pad_len: int) -> 'Sentence':
    """
    Return a new, padded version of the sentence (or self if pad_len is zero).

    Args:
      pad_len: number of tokens to append
    Returns:
      padded sentence
    """
    raise NotImplementedError("must be implemented by subclasses")

  def create_truncated_sent(self, trunc_len: int) -> 'Sentence':
    """
    Create a new, right-truncated version of the sentence (or self if trunc_len is zero).

    Args:
      trunc_len: number of tokens to truncate
    Returns:
      truncated sentence
    """
    raise NotImplementedError("must be implemented by subclasses")

class ReadableSentence(Sentence):
  """
  A base class for sentences based on readable strings.

  Args:
    idx: running sentence number (unique among sentences loaded from the same file, but not across files)
    output_procs: output processors to be applied when calling sent_str()
  """
  def __init__(self, idx: int, output_procs: Sequence[OutputProcessor] = []) -> None:
    super().__init__(idx=idx)
    self.output_procs = output_procs

  def str_tokens(self, **kwargs) -> List[str]:
    """
    Return list of readable string tokens.

    Args:
      **kwargs: should accept arbitrary keyword args

    Returns: list of tokens.
    """
    raise NotImplementedError("must be implemented by subclasses")
  def sent_str(self, **kwargs) -> str:
    """
    Return a single string containing the readable version of the sentence.

    Args:
      **kwargs: should accept arbitrary keyword args

    Returns: readable string
    """
    out_str = " ".join(self.str_tokens(**kwargs))
    # TODO: apply output processors
    return out_str

class ScalarSentence(ReadableSentence):
  """
  A sentence represented by a single integer value, optionally interpreted via a vocab.

  This is useful for classification-style problems.

  Args:
     value: scalar value
     vocab: optional vocab to give different scalar values a string representation.
  """
  def __init__(self, value: int, vocab: Optional[Vocab] = None) -> None:
    self.value = value
    self.vocab = vocab
  def sent_len(self) -> int:
    return 1
  def len_unpadded(self) -> int:
    return 1
  def create_padded_sent(self, pad_len: int) -> 'ScalarSentence':
    if pad_len != 0:
      raise ValueError("ScalarSentence cannot be padded")
    return self
  def create_truncated_sent(self, trunc_len: int) -> 'ScalarSentence':
    if trunc_len != 0:
      raise ValueError("ScalarSentence cannot be truncated")
    return self
  def str_tokens(self, **kwargs) -> List[str]:
    if self.vocab: return [self.vocab[self.value]]
    else: return [str(self.value)]

class CompoundSentence(Sentence):
  """
  A compound sentence contains several sentence objects that present different 'views' on the same data examples.

  Args:
    sents: a list of sentences
  """
  def __init__(self, sents: Sequence[Sentence]) -> None:
    self.sents = sents
  def sent_len(self) -> int:
    return sum(sent.sent_len() for sent in self.sents)
  def len_unpadded(self) -> int:
    return sum(sent.len_unpadded() for sent in self.sents)
  def create_padded_sent(self, pad_len):
    raise ValueError("not supported with CompoundInput, must be called on one of the sub-inputs instead.")
  def create_truncated_sent(self, trunc_len):
    raise ValueError("not supported with CompoundInput, must be called on one of the sub-inputs instead.")


class SimpleSentence(ReadableSentence):
  """
  A simple sentence, represented as a list of tokens

  Args:
    words: list of integer word ids
    vocab: optionally vocab mapping word ids to strings
  """
  PAD_TOKEN = Vocab.ES

  def __init__(self, words: Sequence[int], vocab: Optional[Vocab] = None):
    self.words = words
    self.vocab = vocab

  def __repr__(self):
    return f"SimpleSentence({repr(self.words)})"

  def __str__(self):
    return self.sent_str()

  def __getitem__(self, key):
    ret = self.words[key]
    if isinstance(ret, list):  # support for slicing
      return SimpleSentence(ret, vocab=self.vocab)
    return self.words[key]

  def sent_len(self):
    return len(self.words)

  @functools.lru_cache(maxsize=1)
  def len_unpadded(self):
    return sum(x != Vocab.ES for x in self.words)

  def create_padded_sent(self, pad_len: int) -> 'SimpleSentence':
    if pad_len == 0:
      return self
    # Copy is used to copy all possible annotations
    new_sent = copy.deepcopy(self)
    new_sent.words.extend([SimpleSentence.PAD_TOKEN] * pad_len)
    return new_sent

  def create_truncated_sent(self, trunc_len: int) -> 'SimpleSentence':
    if trunc_len == 0:
      return self
    new_sent = copy.deepcopy(self)
    new_sent.words = self.words[:-trunc_len]
    return new_sent

  def str_tokens(self, exclude_ss_es=True, exclude_unk=False, **kwargs) -> List[str]:
    exclude_set = set()
    if exclude_ss_es:
      exclude_set.add(Vocab.SS, Vocab.ES)
    if exclude_unk: exclude_set.add(self.vocab.unk_token)
    ret_toks =  [w for w in self.words if w not in exclude_set]
    if self.vocab: return [self.vocab[w] for w in ret_toks]
    else: return [str(w) for w in ret_toks]

class ArraySentence(Sentence):
  """
  A sentence based on a numpy array containing a continuous-space vector for each token.

  Args:
    nparr: numpy array of dimension num_tokens x token_size
  """

  def __init__(self, nparr: np.ndarray, padded_len: int = 0) -> None:
    self.nparr = nparr
    self.padded_len = padded_len

  def __getitem__(self, key):
    assert isinstance(key, int)
    return self.nparr.__getitem__(key)

  def sent_len(self):
    # TODO: check, this seems wrong (maybe need a 'transposed' version?)
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def len_unpadded(self):
    return len(self) - self.padded_len

  def create_padded_sent(self, pad_len: int) -> 'ArraySentence':
    if pad_len == 0:
      return self
    new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:, -1], (self.nparr.shape[0], 1)),
                                                      (self.nparr.shape[0], pad_len)), axis=1)
    return ArraySentence(new_nparr, padded_len=self.padded_len + pad_len)

  def create_truncated_sent(self, trunc_len: int) -> 'ArraySentence':
    if trunc_len == 0:
      return self
    new_nparr = np.asarray(self.nparr[:-trunc_len])
    return ArraySentence(new_nparr, padded_len=max(0,self.padded_len - trunc_len))

  def get_array(self):
    return self.nparr

class NbestSentence(Sentence):
  ...
  # TODO