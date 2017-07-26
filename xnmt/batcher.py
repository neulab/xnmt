from __future__ import division, generators

import numpy as np
import six
import vocab

# Shortnames
Vocab = vocab.Vocab

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

  def pack_by_order(self, src, trg, order):
    if self.granularity == 'sent':
      src_ret = [Batch(pad([src[y] for y in order[x:x+self.batch_size]], pad_token=self.src_pad_token)) for x in six.moves.range(0, len(order), self.batch_size)]
      trg_ret = [Batch(pad([trg[y] for y in order[x:x+self.batch_size]], pad_token=self.trg_pad_token)) for x in six.moves.range(0, len(order), self.batch_size)]
    elif self.granularity == 'word':
      src_ret, src_curr = [], []
      trg_ret, trg_curr = [], []
      my_size = 0
      for i in order:
        my_size += self.len_or_zero(src[i]) + self.len_or_zero(trg[i])
        if my_size > self.batch_size:
          src_ret.append(Batch(pad(src_curr, pad_token=self.src_pad_token)))
          trg_ret.append(Batch(pad(trg_curr, pad_token=self.trg_pad_token)))
          my_size = len(src[i]) + len(trg[i])
          src_curr = []
          trg_curr = []
        src_curr.append(src[i])
        trg_curr.append(trg[i])
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    return src_ret, trg_ret

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
    return data
  else:
    return Batch(data)

def is_batched(data):
  return type(data) == Batch

def pad(batch, pad_token=Vocab.ES):
  # Determine the type of batch
  max_len = max(len_or_zero(item) for item in batch)
  return [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch] if max_len > 0 else batch

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

