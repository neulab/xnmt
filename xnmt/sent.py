from typing import List, Optional, Sequence
import functools
import copy

import numpy as np

from xnmt.vocab import Vocab
from xnmt.output import OutputProcessor

class Sentence(object):
  """
  A template class to represent a single data example of any type, used for both model input and output.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    score: a score given to this sentence by a model
  """

  def __init__(self, idx: Optional[int] = None, score: Optional[float] = None) -> None:
    self.idx = idx
    self.score = score

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
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    score: a score given to this sentence by a model
    output_procs: output processors to be applied when calling sent_str()
  """
  def __init__(self, idx: int, score: float = None, output_procs: Sequence[OutputProcessor] = []) -> None:
    super().__init__(idx=idx, score=score)
    self.output_procs = output_procs

  def str_tokens(self, **kwargs) -> List[str]:
    """
    Return list of readable string tokens.

    Args:
      **kwargs: should accept arbitrary keyword args

    Returns: list of tokens.
    """
    raise NotImplementedError("must be implemented by subclasses")
  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    """
    Return a single string containing the readable version of the sentence.

    Args:
      custom_output_procs: if not None, overwrite the sentence's default output processors
      **kwargs: should accept arbitrary keyword args

    Returns: readable string
    """
    out_str = " ".join(self.str_tokens(**kwargs))
    pps = self.output_procs
    if custom_output_procs is not None:
      pps = custom_output_procs
    for pp in pps:
      out_str = pp.process(out_str)
      # TODO: change output processor interface accordingly
    return out_str

class ScalarSentence(ReadableSentence):
  """
  A sentence represented by a single integer value, optionally interpreted via a vocab.

  This is useful for classification-style problems.

  Args:
    value: scalar value
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    vocab: optional vocab to give different scalar values a string representation.
    score: a score given to this sentence by a model
  """
  def __init__(self, value: int, idx: Optional[int] = None, vocab: Optional[Vocab] = None,
               score: Optional[float] = None) -> None:
    super().__init__(idx=idx, score=score)
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
    super().__init__(idx=sents[0].idx)
    self.idx = sents[0].idx
    for s in sents[1:]:
      if s.idx != self.idx:
        raise ValueError("CompoundSentence must contain sentences of consistent idx.")
    self.sents = sents
  def sent_len(self) -> int:
    return sum(sent.sent_len() for sent in self.sents)
  def len_unpadded(self) -> int:
    return sum(sent.len_unpadded() for sent in self.sents)
  def create_padded_sent(self, pad_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")
  def create_truncated_sent(self, trunc_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")


class SimpleSentence(ReadableSentence):
  """
  A simple sentence, represented as a list of tokens

  Args:
    words: list of integer word ids
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    vocab: optionally vocab mapping word ids to strings
    score: a score given to this sentence by a model
    output_procs: output processors to be applied when calling sent_str()
  """
  PAD_TOKEN = Vocab.ES

  def __init__(self, words: Sequence[int], idx: Optional[int] = None, vocab: Optional[Vocab] = None,
               score: Optional[float] = None,
               output_procs: Sequence[OutputProcessor] = []) -> None:
    super().__init__(idx=idx, score=score, output_procs=output_procs)
    self.words = words
    self.vocab = vocab

  def __repr__(self):
    return f"SimpleSentence({repr(self.words)})"

  def __str__(self):
    return self.sent_str()

  def __getitem__(self, key):
    ret = self.words[key]
    if isinstance(ret, list):  # support for slicing
      return SimpleSentence(idx=self.idx, words=ret, vocab=self.vocab)
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

  def str_tokens(self, exclude_ss_es=True, exclude_unk=False, exclude_padded=True, **kwargs) -> List[str]:
    exclude_set = set()
    if exclude_ss_es:
      exclude_set.add(Vocab.SS)
      exclude_set.add(Vocab.ES)
    if exclude_unk: exclude_set.add(self.vocab.unk_token)
    # TODO: exclude padded if requested (i.e., all </s> tags except for the first)
    ret_toks =  [w for w in self.words if w not in exclude_set]
    if self.vocab: return [self.vocab[w] for w in ret_toks]
    else: return [str(w) for w in ret_toks]

class ArraySentence(Sentence):
  """
  A sentence based on a numpy array containing a continuous-space vector for each token.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    nparr: numpy array of dimension num_tokens x token_size
    padded_len: how many padded tokens are contained in the given nparr
    score: a score given to this sentence by a model
  """

  def __init__(self, nparr: np.ndarray, idx: Optional[int] = None, padded_len: int = 0,
               score: Optional[float] = None) -> None:
    super().__init__(idx=idx, score=score)
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

class NbestSentence(SimpleSentence):
  """
  Output in the context of an nbest list.

  Args:
    base_sent: The base sent object
    nbest_id: The sentence id in the nbest list
    print_score: If True, print nbest_id, score, content separated by ``|||``. If False, drop the score.
  """
  def __init__(self, base_sent: SimpleSentence, nbest_id: int, print_score: bool = False) -> None:
    super().__init__(words=base_sent.words, vocab=base_sent.vocab, score=base_sent.score)
    self.base_output = base_sent
    self.nbest_id = nbest_id
    self.print_score = print_score
  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    content_str = super().sent_str(custom_output_procs=custom_output_procs, **kwargs)
    return self._make_nbest_entry(content_str=content_str)
  def _make_nbest_entry(self, content_str: str) -> str:
    entries = [str(self.nbest_id), content_str]
    if self.print_score:
      entries.insert(1, str(self.base_output.score))
    return " ||| ".join(entries)