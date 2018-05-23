import math
import random
import numpy as np
import dynet as dy
from xnmt.vocab import Vocab
from xnmt.persistence import serializable_init, Serializable

class Batch(list):
  """
  A class containing a minibatch of things.

  This class behaves like a Python list, but adds semantics that the contents form a (mini)batch of things.
  An optional mask can be specified to indicate padded parts of the inputs.
  Should be treated as an immutable object.
  
  Args:
    batch_list (list): list of things
    mask (Mask): optional mask when  batch contains items of unequal size
  """
  def __init__(self, batch_list, mask=None):
    super(Batch, self).__init__(batch_list)
    self.mask = mask

class Mask(object):
  """
  A mask specifies padded parts in a sequence or batch of sequences.

  Masks are represented as numpy array of dimensions batchsize x seq_len, with parts
  belonging to the sequence set to 0, and parts that should be masked set to 1
  
  Args:
    np_arr: numpy array
  """
  def __init__(self, np_arr):
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
    batch_size (int): batch size
    granularity (str): 'sent' or 'word'
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               pad_src_to_multiple=1):
    self.batch_size = batch_size
    self.src_pad_token = src_pad_token
    self.trg_pad_token = trg_pad_token
    self.granularity = granularity
    self.pad_src_to_multiple = pad_src_to_multiple

  def is_random(self):
    """
    Returns:
      True if there is some randomness in the batching process, False otherwise.
    """
    return False

  def add_single_batch(self, src_curr, trg_curr, src_ret, trg_ret):
    src_id, src_mask = pad(src_curr, pad_token=self.src_pad_token, pad_to_multiple=self.pad_src_to_multiple)
    src_ret.append(Batch(src_id, src_mask))
    if trg_ret is not None:
      trg_id, trg_mask = pad(trg_curr, pad_token=self.trg_pad_token)
      trg_ret.append(Batch(trg_id, trg_mask))

  def pack_by_order(self, src, trg, order):
    src_ret, src_curr = [], []
    trg_ret, trg_curr = [], []
    if self.granularity == 'sent':
      for x in range(0, len(order), self.batch_size):
        self.add_single_batch([src[y] for y in order[x:x+self.batch_size]], [trg[y] for y in order[x:x+self.batch_size]], src_ret, trg_ret)
    elif self.granularity == 'word':
      my_size = 0
      for i in order:
        my_size += len_or_zero(src[i]) + len_or_zero(trg[i])
        if my_size > self.batch_size and len(src_curr)>0:
          self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret)
          my_size = len(src[i]) + len(trg[i])
          src_curr = []
          trg_curr = []
        src_curr.append(src[i])
        trg_curr.append(trg[i])
      self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret)
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    return src_ret, trg_ret

class InOrderBatcher(Batcher, Serializable):
  """
  A class to create batches in order of the original corpus.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!InOrderBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               pad_src_to_multiple=1):
    super(InOrderBatcher, self).__init__(batch_size, src_pad_token=src_pad_token,
                                         trg_pad_token=trg_pad_token,
                                         pad_src_to_multiple=pad_src_to_multiple)

  def pack(self, src, trg):
    order = list(range(len(src)))
    return self.pack_by_order(src, trg, order)

class ShuffleBatcher(Batcher):
  """
  A template class to create batches through randomly shuffling without sorting.
  """

  def pack(self, src, trg):
    order = list(range(len(src)))
    np.random.shuffle(order)
    return self.pack_by_order(src, trg, order)

  def is_random(self):
    return True

class SortBatcher(Batcher):
  """
  A template class to create batches through bucketing sentence length.
  """
  __tiebreaker_eps = 1.0e-7

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES,
               trg_pad_token=Vocab.ES, sort_key=lambda x: len(x[0]),
               break_ties_randomly=True, pad_src_to_multiple=1):
    super(SortBatcher, self).__init__(batch_size, granularity=granularity,
                                      src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                      pad_src_to_multiple=pad_src_to_multiple)
    self.sort_key = sort_key
    self.break_ties_randomly = break_ties_randomly

  def pack(self, src, trg):
    if self.break_ties_randomly:
      order = np.argsort([self.sort_key(x) + random.uniform(-SortBatcher.__tiebreaker_eps, SortBatcher.__tiebreaker_eps) for x in zip(src,trg)])
    else:
      order = np.argsort([self.sort_key(x) for x in zip(src,trg)])
    return self.pack_by_order(src, trg, order)

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
  if type(data) == Batch and mask is None:
    ret = data
  else:
    ret = Batch(data, mask)
  return ret

def is_batched(data):
  """
  Check whether some data is batched.

  Args:
    data: data to check

  Returns:
    True iff data is batched.
  """
  return type(data) == Batch

def pad(batch, pad_token=Vocab.ES, pad_to_multiple=1):
  """
  Apply padding to sentences in a batch.

  Args:
    batch: batch of sentences
    pad_token: token to pad with
    pad_to_multiple (int): pad sentences so their length is a multiple of this integer.

  Returns:
    Tuple: list of padded items and a corresponding batched mask.
  """
  max_len = max(len_or_zero(item) for item in batch)
  if max_len % pad_to_multiple != 0:
    max_len += pad_to_multiple - (max_len % pad_to_multiple)
  min_len = min(len_or_zero(item) for item in batch)
  if min_len == max_len:
    return batch, None
  masks = np.zeros([len(batch), max_len])
  for i, v in enumerate(batch):
    for j in range(len_or_zero(v), max_len):
      masks[i,j] = 1.0
  padded_items = [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch]
  return padded_items, Mask(masks)

def len_or_zero(val):
  return len(val) if hasattr(val, '__len__') else 0

class SrcBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               break_ties_randomly:bool=True, pad_src_to_multiple=1):
    super(SrcBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[0]), granularity='sent',
                                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                     break_ties_randomly=break_ties_randomly,
                                     pad_src_to_multiple=pad_src_to_multiple)

class TrgBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               break_ties_randomly:bool=True, pad_src_to_multiple=1):
    super(TrgBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[1]), granularity='sent',
                                     src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                     break_ties_randomly=break_ties_randomly,
                                     pad_src_to_multiple=pad_src_to_multiple)

class SrcTrgBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len, then trg len.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcTrgBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               break_ties_randomly:bool=True, pad_src_to_multiple=1):
    super(SrcTrgBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]),
                                        granularity='sent',
                                        src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                        break_ties_randomly=break_ties_randomly,
                                        pad_src_to_multiple=pad_src_to_multiple)

class TrgSrcBatcher(SortBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len, then src len.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgSrcBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               break_ties_randomly:bool=True, pad_src_to_multiple=1):
    super(TrgSrcBatcher, self).__init__(batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]),
                                        granularity='sent',
                                        src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                        break_ties_randomly=break_ties_randomly,
                                        pad_src_to_multiple=pad_src_to_multiple)

class SentShuffleBatcher(ShuffleBatcher, Serializable):
  """
  A batcher that creates fixed-size batches or random order.
  
  Args:
    batch_size (int): batch size
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SentShuffleBatcher"

  @serializable_init
  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               pad_src_to_multiple=1):
    super(SentShuffleBatcher, self).__init__(batch_size, granularity='sent', src_pad_token=src_pad_token,
                                             trg_pad_token=trg_pad_token, pad_src_to_multiple=pad_src_to_multiple)

class WordShuffleBatcher(ShuffleBatcher, Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.
  
  Args:
    words_per_batch (int): number of src+trg words in each batch
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordShuffleBatcher"

  @serializable_init
  def __init__(self, words_per_batch, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES,
               pad_src_to_multiple=1):
    super(WordShuffleBatcher, self).__init__(words_per_batch, granularity='word', src_pad_token=src_pad_token,
                                             trg_pad_token=trg_pad_token, pad_src_to_multiple=pad_src_to_multiple)

class WordSortBatcher(SortBatcher):
  """
  Base class for word sort-based batchers
  """
  def __init__(self, words_per_batch, avg_batch_size, sort_key,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly=True,
               pad_src_to_multiple=1):
    # Sanity checks
    if words_per_batch and avg_batch_size:
      raise ValueError("words_per_batch and avg_batch_size are mutually exclusive.")
    elif words_per_batch is None and avg_batch_size is None:
      raise ValueError("either words_per_batch or avg_batch_size must be specified.")

    super(WordSortBatcher, self).__init__(words_per_batch, sort_key=sort_key, granularity='word',
                                          src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                          break_ties_randomly=break_ties_randomly,
                                          pad_src_to_multiple=pad_src_to_multiple)
    self.avg_batch_size = avg_batch_size

class WordSrcBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by src len.
  
  Args:
    words_per_batch (int): number of src+trg words in each batch
    avg_batch_size (number): avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcBatcher"

  @serializable_init
  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple=1):
    super(WordSrcBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[0]),
                                         src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                         break_ties_randomly=break_ties_randomly,
                                         pad_src_to_multiple=pad_src_to_multiple)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super(WordSrcBatcher, self).pack_by_order(src, trg, order)

class WordTrgBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by trg len.
  
  Args:
    words_per_batch (int): number of src+trg words in each batch
    avg_batch_size (number): avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgBatcher"

  @serializable_init
  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple=1):
    super(WordTrgBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[1]),
                                         src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                         break_ties_randomly=break_ties_randomly,
                                         pad_src_to_multiple=pad_src_to_multiple)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super(WordTrgBatcher, self).pack_by_order(src, trg, order)

class WordSrcTrgBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by src len, then trg len.
  
  Args:
    words_per_batch (int): number of src+trg words in each batch
    avg_batch_size (number): avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcTrgBatcher"

  @serializable_init
  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple=1):
    super(WordSrcTrgBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]),
                                            src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                            break_ties_randomly=break_ties_randomly,
                                            pad_src_to_multiple=pad_src_to_multiple)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super(WordSrcTrgBatcher, self).pack_by_order(src, trg, order)

class WordTrgSrcBatcher(WordSortBatcher, Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by trg len, then src len.
  
  Args:
    words_per_batch (int): number of src+trg words in each batch
    avg_batch_size (number): avg number of sentences in each batch (if words_per_batch not given)
    src_pad_token: token used to pad on source side
    trg_pad_token: token used to pad on target side
    break_ties_randomly (bool): if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple (int): pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgSrcBatcher"

  @serializable_init
  def __init__(self, words_per_batch=None, avg_batch_size=None,
               src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES, break_ties_randomly:bool=True,
               pad_src_to_multiple=1):
    super(WordTrgSrcBatcher, self).__init__(words_per_batch, avg_batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]),
                                            src_pad_token=src_pad_token, trg_pad_token=trg_pad_token,
                                            break_ties_randomly=break_ties_randomly,
                                            pad_src_to_multiple=pad_src_to_multiple)

  def pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([len(s) for s in src]) + sum([len(s) for s in trg])) / len(src) * self.avg_batch_size
    return super(WordTrgSrcBatcher, self).pack_by_order(src, trg, order)

