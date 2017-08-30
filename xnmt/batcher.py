from __future__ import division, generators

import numpy as np
import six
import xnmt.vocab

# Shortnames
Vocab = xnmt.vocab.Vocab

class Batch(list):
  """
  A marker class to indicate a batch.
  """
  pass


class Batcher(object):
  """
  A template class to convert a list of sents to several batches of sents.
  """

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    self.batch_size = batch_size
    self.src_pad_token = src_pad_token
    self.trg_pad_token = trg_pad_token
    self.granularity = granularity

  def is_random(self):
    """
    :returns: True if there is some randomness in the batching process, False otherwise. Defaults to false.
    """
    return False

  def add_single_batch(self, src_curr, trg_curr, src_ret, trg_ret, src_masks, trg_masks):
     src_id, src_mask = pad(src_curr, pad_token=self.src_pad_token)
     src_ret.append(Batch(src_id))
     src_masks.append(src_mask)
     trg_id, trg_mask = pad(trg_curr, pad_token=self.trg_pad_token)
     trg_ret.append(Batch(trg_id))
     trg_masks.append(trg_mask)

  def pack_by_order(self, src, trg, order):
    src_ret, src_curr, src_masks = [], [], []
    trg_ret, trg_curr, trg_masks = [], [], []
    if self.granularity == 'sent':
      for x in six.moves.range(0, len(order), self.batch_size):
        self.add_single_batch([src[y] for y in order[x:x+self.batch_size]], [trg[y] for y in order[x:x+self.batch_size]], src_ret, trg_ret, src_masks, trg_masks)
    elif self.granularity == 'word':
      my_size = 0
      for i in order:
        my_size += len_or_zero(src[i]) + len_or_zero(trg[i])
        if my_size > self.batch_size:
          self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret, src_masks, trg_masks)
          my_size = len(src[i]) + len(trg[i])
          src_curr = []
          trg_curr = []
        src_curr.append(src[i])
        trg_curr.append(trg[i])
      self.add_single_batch(src_curr, trg_curr, src_ret, trg_ret, src_masks, trg_masks)
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    return src_ret, src_masks, trg_ret, trg_masks

class InOrderBatcher(Batcher):
  """
  A class to create batches in order of the original corpus.
  """

  def pack(self, src, trg):
    order = list(range(len(src)))
    return self.pack_by_order(src, trg, order)

class ShuffleBatcher(Batcher):
  """
  A class to create batches through randomly shuffling without sorting.
  """

  def pack(self, src, trg):
    order = list(range(len(src)))
    np.random.shuffle(order)
    return self.pack_by_order(src, trg, order)

  def is_random(self):
    return True

class SortBatcher(Batcher):
  """
  A template class to create batches through bucketing sent length.
  """

  def __init__(self, batch_size, granularity='sent', src_pad_token=Vocab.ES,
               trg_pad_token=Vocab.ES, sort_key=lambda x: len(x[0])):
    super(SortBatcher, self).__init__(batch_size, granularity=granularity,
                                      src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    self.sort_key = sort_key

  def pack(self, src, trg):
    order = np.argsort([self.sort_key(x) for x in six.moves.zip(src,trg)])
    return self.pack_by_order(src, trg, order)

# Module level functions
def mark_as_batch(data):
  if type(data) == Batch:
    ret = data
  else:
    ret = Batch(data)
  return ret
    

def is_batched(data):
  return type(data) == Batch

def pad(batch, pad_token=Vocab.ES):
  max_len = max(len_or_zero(item) for item in batch)
  min_len = min(len_or_zero(item) for item in batch)
  if min_len == max_len:
    return batch, None
  masks = np.zeros([len(batch), max_len])
  for i, v in enumerate(batch):
    for j in range(len_or_zero(v), max_len):
      masks[i,j] = 1.0
  padded_items = [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch]
  return padded_items, masks

def len_or_zero(val):
  return len(val) if hasattr(val, '__len__') else 0

def from_spec(batcher_spec, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
  if batcher_spec == 'src':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[0]), granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'trg':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[1]), granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'src_trg':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]), granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'trg_src':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]), granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'shuffle':
    return ShuffleBatcher(batch_size, granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'inorder':
    return InOrderBatcher(batch_size, granularity='sent', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'word_src':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[0]), granularity='word', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'word_trg':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[1]), granularity='word', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'word_src_trg':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]), granularity='word', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'word_trg_src':
    return SortBatcher(batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]), granularity='word', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  elif batcher_spec == 'word_shuffle':
    return ShuffleBatcher(batch_size, granularity='word', src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
  else:
    raise RuntimeError("Illegal batcher specification {}".format(batcher_spec))

