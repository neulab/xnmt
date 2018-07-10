import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import dynet as dy

from xnmt.vocab import Vocab
from xnmt.persistence import serializable_init, Serializable
import xnmt.expression_sequence
from xnmt import lstm
import xnmt.input

class Batch(ABC):
  """
  An abstract base class for minibatches of things.
  """
  @abstractmethod
  def batch_size(self) -> int:
    raise NotImplementedError()
  def sent_len(self) -> int:
    raise NotImplementedError()

class ListBatch(list, Batch):
  """
  A class containing a minibatch of things.

  This class behaves like a Python list, but adds semantics that the contents form a (mini)batch of things.
  An optional mask can be specified to indicate padded parts of the inputs.
  Should be treated as an immutable object.
  
  Args:
    batch_elements: list of things
    mask: optional mask when  batch contains items of unequal size
  """
  def __init__(self, batch_elements: list, mask: 'Mask'=None) -> None:
    assert len(batch_elements)>0
    super().__init__(batch_elements)
    self.mask = mask
  def batch_size(self) -> int: return super().__len__()
  def sent_len(self) -> int: return self[0].sent_len()
  def __len__(self):
    warnings.warn("use of ListBatch.__len__() is discouraged, use ListBatch.batch_size() "
                  "[or ListBatch.sent_len()] instead.", DeprecationWarning)
  def __getitem__(self, key):
    ret = super().__getitem__(key)
    if isinstance(key, slice):
      ret = ListBatch(ret)
    return ret


class CompoundBatch(Batch):
  """
  A compound batch contains several parallel batches.

  Args:
    *batch_elements: one or several batches
  """

  def __init__(self, *batch_elements: Batch):
    assert len(batch_elements) > 0
    self.batches = batch_elements

  def batch_size(self):
    return self.batches[0].batch_size()

  def sent_len(self):
    return sum(b.sent_len() for b in self.batches)

  def __iter__(self):
    for i in self.batch_size():
      yield xnmt.input.CompoundInput([b[i] for b in self.batches])

  def __getitem__(self, key):
    if isinstance(key, int):
      return xnmt.input.CompoundInput([b[key] for b in self.batches])
    else:
      assert isinstance(key, slice)
      sel_batches = [b[key] for b in self.batches]
      return CompoundBatch(sel_batches)


class Mask(object):
  """
  A mask specifies padded parts in a sequence or batch of sequences.

  Masks are represented as numpy array of dimensions batchsize x seq_len, with parts
  belonging to the sequence set to 0, and parts that should be masked set to 1
  
  Args:
    np_arr: numpy array
  """
  def __init__(self, np_arr: np.ndarray):
    self.np_arr = np_arr

  def __len__(self):
    return self.np_arr.shape[1]

  def batch_size(self):
    return self.np_arr.shape[0]

  def reversed(self):
    return Mask(self.np_arr[:,::-1])

  def add_to_tensor_expr(self, tensor_expr, multiplicator=None):
    # TODO: might cache these expressions to save memory
    if np.count_nonzero(self.np_arr) == 0:
      return tensor_expr
    else:
      if multiplicator is not None:
        mask_expr = dy.inputTensor(np.expand_dims(self.np_arr.transpose(), axis=1) * multiplicator, batched=True)
      else:
        mask_expr = dy.inputTensor(np.expand_dims(self.np_arr.transpose(), axis=1), batched=True)
      return tensor_expr + mask_expr

  def lin_subsampled(self, reduce_factor=None, trg_len=None):
    if reduce_factor:
      return Mask(np.array([[self.np_arr[b,int(i*reduce_factor)] for i in range(int(math.ceil(len(self)/float(reduce_factor))))] for b in range(self.batch_size())]))
    else:
      return Mask(np.array([[self.np_arr[b,int(i*len(self)/float(trg_len))] for i in range(trg_len)] for b in range(self.batch_size())]))

  def cmult_by_timestep_expr(self, expr, timestep, inverse=False):
    # TODO: might cache these expressions to save memory
    """
    Args:
      expr: a dynet expression corresponding to one timestep
      timestep: index of current timestep
      inverse: True will keep the unmasked parts, False will zero out the unmasked parts
    """
    if inverse:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == 0:
        return expr
      mask_exp = dy.inputTensor((1.0 - self.np_arr)[:,timestep:timestep+1].transpose(), batched=True)
    else:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == self.np_arr[:,timestep:timestep+1].size:
        return expr
      mask_exp = dy.inputTensor(self.np_arr[:,timestep:timestep+1].transpose(), batched=True)
    return dy.cmult(expr, mask_exp)

  def get_active_one_mask(self):
    return 1 - self.np_arr

class Batcher(object):
  """
  A template class to convert a list of sentences to several batches of sentences.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
    sort_within_by_trg_len: whether to sort by reverse trg len inside a batch
  """

  def __init__(self, batch_size: int, granularity: str = 'sent', src_pad_token: Any = Vocab.ES,
               trg_pad_token: Any = Vocab.ES, pad_src_to_multiple: int = 1,
               sort_within_by_trg_len: bool = True) -> None:
    self.batch_size = batch_size
    self.src_pad_token = src_pad_token
    self.trg_pad_token = trg_pad_token
    self.granularity = granularity
    self.pad_src_to_multiple = pad_src_to_multiple
    self.sort_within_by_trg_len = sort_within_by_trg_len

  def is_random(self):
    """
    Returns:
      True if there is some randomness in the batching process, False otherwise.
    """
    return False

  def create_single_batch(self, src_sents: Sequence[xnmt.input.Input],
                          trg_sents: Optional[Sequence[xnmt.input.Input]] = None,
                          sort_by_trg_len: bool = False) -> Union[Batch, Tuple[Batch]]:
    """
    Create a single batch, either source-only or source-and-target.

    Args:
      src_sents: list of source-side inputs
      trg_sents: optional list of target-side inputs
      sort_by_trg_len: if True (and targets are specified), sort source- and target batches by target length

    Returns:
      a tuple of batches if targets were given, otherwise a single batch
    """
    if trg_sents is not None and sort_by_trg_len:
      src_sents, trg_sents = zip(*sorted(zip(src_sents, trg_sents), key=lambda x: x[1].sent_len(), reverse=True))
    src_batch = pad(src_sents, pad_token=self.src_pad_token, pad_to_multiple=self.pad_src_to_multiple)
    if trg_sents is None:
      return src_batch
    else:
      trg_batch = pad(trg_sents, pad_token=self.trg_pad_token)
      return src_batch, trg_batch

  def _add_single_batch(self, src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=False):
    if trg_curr:
      src_batch, trg_batch = self.create_single_batch(src_curr, trg_curr, sort_by_trg_len)
      trg_ret.append(trg_batch)
    else:
      src_batch = self.create_single_batch(src_curr, trg_curr, sort_by_trg_len)
    src_ret.append(src_batch)

  def _pack_by_order(self, src: Sequence[xnmt.input.Input], trg: Optional[Sequence[xnmt.input.Input]],
                     order: Sequence[int]) -> Tuple[Sequence[Batch], Sequence[Batch]]:
    """
    Pack batches by given order.

    Trg is optional for the case of self.granularity == 'sent'

    Args:
      src: src-side inputs
      trg: trg-side inputs
      order: order of inputs

    Returns:
      If trg is given: tuple of src / trg batches; Otherwise: only src batches
    """
    src_ret, src_curr = [], []
    trg_ret, trg_curr = [], []
    if self.granularity == 'sent':
      for x in range(0, len(order), self.batch_size):
        src_selected = [src[y] for y in order[x:x + self.batch_size]]
        if trg:
          trg_selected = [trg[y] for y in order[x:x + self.batch_size]]
        else: trg_selected = None
        self._add_single_batch(src_selected,
                               trg_selected,
                               src_ret, trg_ret,
                               sort_by_trg_len=self.sort_within_by_trg_len)
    elif self.granularity == 'word':
      max_src, max_trg = 0, 0
      for i in order:
        max_src = max(_len_or_zero(src[i]), max_src)
        max_trg = max(_len_or_zero(trg[i]), max_trg)
        if (max_src + max_trg) * (len(src_curr) + 1) > self.batch_size and len(src_curr) > 0:
          self._add_single_batch(src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=self.sort_within_by_trg_len)
          max_src = _len_or_zero(src[i])
          max_trg = _len_or_zero(trg[i])
          src_curr = [src[i]]
          trg_curr = [trg[i]]
        else:
          src_curr.append(src[i])
          trg_curr.append(trg[i])
      self._add_single_batch(src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=self.sort_within_by_trg_len)
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    if trg:
      return src_ret, trg_ret
    else:
      return src_ret

  def pack(self, src: Sequence[xnmt.input.Input], trg: Sequence[xnmt.input.Input]) \
          -> Tuple[Sequence[Batch], Sequence[Batch]]:
    """
    Create a list of src/trg batches based on provided src/trg inputs.

    Args:
      src: list of src-side inputs
      trg: list of trg-side inputs

    Returns:
      tuple of lists of src and trg batches
    """
    raise NotImplementedError("must be implemented by subclasses")

class InOrderBatcher(Batcher, Serializable):
  """
  A class to create batches in order of the original corpus, both across and within batches.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!InOrderBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, src_pad_token=src_pad_token,
                     trg_pad_token=trg_pad_token,
                     pad_src_to_multiple=pad_src_to_multiple,
                     sort_within_by_trg_len=False)

  def pack(self, src: Sequence[xnmt.input.Input], trg: Optional[Sequence[xnmt.input.Input]]) \
          -> Tuple[Sequence[Batch], Sequence[Batch]]:
    """
    Pack batches. Unlike other batches, the trg sentences are optional.

    Args:
      src: list of src-side inputs
      trg: optional list of trg-side inputs

    Returns:
      src batches if trg was not given; tuple of src batches and trg batches if trg was given
    """
    order = list(range(len(src)))
    return self._pack_by_order(src, trg, order)

class ShuffleBatcher(Batcher):
  """
  A template class to create batches through randomly shuffling without sorting.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """

  def __init__(self, batch_size: int, granularity: str = 'sent', src_pad_token: Any = Vocab.ES,
               trg_pad_token: Any = Vocab.ES, pad_src_to_multiple: int = 1) -> None:
    super(batch_size=batch_size, granularity=granularity, src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
          pad_src_to_multiple=pad_src_to_multiple, sort_within_by_trg_len=True)

  def pack(self, src, trg):
    order = list(range(len(src)))
    np.random.shuffle(order)
    return self._pack_by_order(src, trg, order)

  def is_random(self):
    return True

class SortBatcher(Batcher):
  """
  A template class to create batches through bucketing sentence length.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  __tiebreaker_eps = 1.0e-7

  def __init__(self, batch_size: int, granularity: str = 'sent', src_pad_token: Any = Vocab.ES,
               trg_pad_token: Any = Vocab.ES, sort_key: Callable = lambda x: x[0].sent_len(),
               break_ties_randomly=True, pad_src_to_multiple=1) -> None:
    super().__init__(batch_size, granularity=granularity,
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     pad_src_to_multiple=pad_src_to_multiple,
                     sort_within_by_trg_len=True)
    self.sort_key = sort_key
    self.break_ties_randomly = break_ties_randomly

  def pack(self, src, trg):
    if self.break_ties_randomly:
      order = np.argsort([self.sort_key(x) + random.uniform(-SortBatcher.__tiebreaker_eps, SortBatcher.__tiebreaker_eps) for x in zip(src,trg)])
    else:
      order = np.argsort([self.sort_key(x) for x in zip(src,trg)])
    return self._pack_by_order(src, trg, order)

  def is_random(self):
    return self.break_ties_randomly

# Module level functions
def mark_as_batch(data, mask=None):
  """
  Mark a sequence of items as batch

  Args:
    data: sequence of things
    mask: optional mask

  Returns: a batch of things
  """
  if isinstance(data, Batch) and mask is None:
    ret = data
  else:
    ret = ListBatch(data, mask)
  return ret

def is_batched(data):
  """
  Check whether some data is batched.

  Args:
    data: data to check

  Returns:
    True iff data is batched.
  """
  return isinstance(data, Batch)

def pad(batch: Sequence, pad_token=Vocab.ES, pad_to_multiple=1) -> Batch:
  """
  Apply padding to sentences in a batch.

  Args:
    batch: batch of sentences
    pad_token: token to pad with
    pad_to_multiple (int): pad sentences so their length is a multiple of this integer.

  Returns:
    batch containing padded items and a corresponding batch mask.
  """
  if isinstance(list(batch)[0], xnmt.input.CompoundInput):
    ret = []
    for compound_i in range(len(batch[0].inputs)):
      ret.append(
        pad(tuple(inp.inputs[compound_i] for inp in batch), pad_token=pad_token, pad_to_multiple=pad_to_multiple))
    return CompoundBatch(*ret)
  max_len = max(_len_or_zero(item) for item in batch)
  if max_len % pad_to_multiple != 0:
    max_len += pad_to_multiple - (max_len % pad_to_multiple)
  min_len = min(_len_or_zero(item) for item in batch)
  if min_len == max_len:
    return ListBatch(batch, mask=None)
  masks = np.zeros([len(batch), max_len])
  for i, v in enumerate(batch):
    for j in range(_len_or_zero(v), max_len):
      masks[i,j] = 1.0
  padded_items = [item.get_padded_sent(pad_token, max_len - item.sent_len()) for item in batch]
  return ListBatch(padded_items, mask=Mask(masks))

def _len_or_zero(val):
  return val.sent_len() if hasattr(val, 'sent_len') else len(val) if hasattr(val, '__len__') else 0

class SrcBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               break_ties_randomly: bool = True, pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[0].sent_len(), granularity='sent',
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

class TrgBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               break_ties_randomly: bool = True, pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[1].sent_len(), granularity='sent',
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

class SrcTrgBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len, then trg len.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcTrgBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               break_ties_randomly: bool = True, pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[0].sent_len() + 1.0e-6 * len(x[1]),
                     granularity='sent',
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

class TrgSrcBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len, then src len.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgSrcBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               break_ties_randomly: bool = True, pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[1].sent_len() + 1.0e-6 * len(x[0]),
                     granularity='sent',
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

class SentShuffleBatcher(ShuffleBatcher, Serializable):
  """
  A batcher that creates fixed-size batches of random order.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    batch_size: batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SentShuffleBatcher"

  @serializable_init
  def __init__(self, batch_size: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, granularity='sent', src_pad_token=src_pad_token,
                     trg_pad_token=trg_pad_token, pad_src_to_multiple=pad_src_to_multiple)

class WordShuffleBatcher(ShuffleBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.
  
  Args:
    words_per_batch: number of src+trg words in each batch
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordShuffleBatcher"

  @serializable_init
  def __init__(self, words_per_batch: int, src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, granularity='word', src_pad_token=src_pad_token,
                     trg_pad_token=trg_pad_token, pad_src_to_multiple=pad_src_to_multiple)

class WordSortBatcher(SortBatcher):
  """
  Base class for word sort-based batchers.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    sort_key:
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """

  def __init__(self, words_per_batch: Optional[int], avg_batch_size: Optional[Union[int,float]], sort_key: Callable,
               src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES, break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    # Sanity checks
    if words_per_batch and avg_batch_size:
      raise ValueError("words_per_batch and avg_batch_size are mutually exclusive.")
    elif words_per_batch is None and avg_batch_size is None:
      raise ValueError("either words_per_batch or avg_batch_size must be specified.")

    super().__init__(words_per_batch, sort_key=sort_key, granularity='word',
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)
    self.avg_batch_size = avg_batch_size

class WordSrcBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcBatcher"

  @serializable_init
  def __init__(self, words_per_batch:Optional[int]=None, avg_batch_size:Optional[Union[int,float]]=None,
               src_pad_token:Any=Vocab.ES, trg_pad_token:Any=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple:int=1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[0].sent_len(),
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)

class WordTrgBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgBatcher"

  @serializable_init
  def __init__(self, words_per_batch:Optional[int]=None, avg_batch_size:Optional[Union[int,float]]=None,
               src_pad_token:Any=Vocab.ES, trg_pad_token:Any=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple:int=1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[1].sent_len(),
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)

class WordSrcTrgBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by src len, then trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcTrgBatcher"

  @serializable_init
  def __init__(self, words_per_batch: Optional[int] = None, avg_batch_size: Optional[Union[int, float]] = None,
               src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES, break_ties_randomly: bool = True,
               pad_src_to_multiple: bool = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[0].sent_len() + 1.0e-6 * x[1].sent_len(),
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)

class WordTrgSrcBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by trg len, then src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgSrcBatcher"

  @serializable_init
  def __init__(self, words_per_batch: Optional[int] = None, avg_batch_size: Optional[Union[int, float]] = None,
               src_pad_token: Any = Vocab.ES, trg_pad_token: Any = Vocab.ES, break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[1].sent_len() + 1.0e-6 * x[0].sent_len(),
                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)

def truncate_batches(*xl: Union[dy.Expression, Batch, Mask, lstm.UniLSTMState]) \
        -> Sequence[Union[dy.Expression, Batch, Mask, lstm.UniLSTMState]]:
  """
  Truncate a list of batched items so that all items have the batch size of the input with the smallest batch size.

  Inputs can be of various types and would usually correspond to a single time step.
  Assume that the batch elements with index 0 correspond across the inputs, so that batch elements will be truncated
  from the top, i.e. starting with the highest-indexed batch elements.
  Masks are not considered even if attached to a input of :class:`Batch` type.

  Args:
    *xl: batched timesteps of various types

  Returns:
    Copies of the inputs, truncated to consistent batch size.
  """
  batch_sizes = []
  for x in xl:
    if isinstance(x, dy.Expression) or isinstance(x, xnmt.expression_sequence.ExpressionSequence):
      batch_sizes.append(x.dim()[1])
    elif isinstance(x, Batch):
      batch_sizes.append(len(x))
    elif isinstance(x, Mask):
      batch_sizes.append(x.batch_size())
    elif isinstance(x, lstm.UniLSTMState):
      batch_sizes.append(x.output().dim()[1])
    else:
      raise ValueError(f"unsupported type {type(x)}")
    assert batch_sizes[-1] > 0
  ret = []
  for i, x in enumerate(xl):
    if batch_sizes[i] > min(batch_sizes):
      if isinstance(x, dy.Expression) or isinstance(x, xnmt.expression_sequence.ExpressionSequence):
        ret.append(x[tuple([slice(None)]*len(x.dim()[0]) + [slice(min(batch_sizes))])])
      elif isinstance(x, Batch):
        ret.append(mark_as_batch(x[:min(batch_sizes)]))
      elif isinstance(x, Mask):
        ret.append(Mask(x.np_arr[:min(batch_sizes)]))
      elif isinstance(x, lstm.UniLSTMState):
        ret.append(x[:,:min(batch_sizes)])
      else:
        raise ValueError(f"unsupported type {type(x)}")
    else:
      ret.append(x)
  return ret
