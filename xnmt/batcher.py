from __future__ import division, generators

import numpy as np
import six
from collections import defaultdict
from vocab import Vocab
from collections import OrderedDict


class Batch(list):
  """
  A marker class to indicate a batch.
  """
  pass


class Batcher:
  """
  A template class to convert a list of sents to several batches of sents.
  """

  def __init__(self, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    self.batch_size = batch_size
    self.src_pad_token = src_pad_token
    self.trg_pad_token = trg_pad_token

  @staticmethod
  def mark_as_batch(data):
    if type(data) == Batch:
      return data
    else:
      return Batch(data)

  @staticmethod
  def is_batched(data):
    return type(data) == Batch

  @staticmethod
  def pad(batch, pad_token=Vocab.ES):
    # Determine the type of batch
    max_len = max(len(item) for item in batch)
    return [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch]

  @staticmethod
  def from_spec(batcher_spec, batch_size, src_pad_token=Vocab.ES, trg_pad_token=Vocab.ES):
    if batcher_spec == 'src':
      return SortBatcher(batch_size, sort_key=lambda x: len(x[0]), src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    elif batcher_spec == 'trg':
      return SortBatcher(batch_size, sort_key=lambda x: len(x[1]), src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    elif batcher_spec == 'src_trg':
      return SortBatcher(batch_size, sort_key=lambda x: len(x[0])+1.0e-6*len(x[1]), src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    elif batcher_spec == 'trg_src':
      return SortBatcher(batch_size, sort_key=lambda x: len(x[1])+1.0e-6*len(x[0]), src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    elif batcher_spec == 'shuffle':
      return ShuffleBatcher(batch_size, src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)

class SentenceBatcher(Batcher):
  """
  A batcher that separates into an equal number of sentences.
  """

  def pack_by_order(self, src, trg, order):

    src_ret = [Batch(Batcher.pad([src[y] for y in order[x:x+self.batch_size]], pad_token=self.src_pad_token)) for x in six.moves.range(0, len(order), self.batch_size)]
    trg_ret = [Batch(Batcher.pad([trg[y] for y in order[x:x+self.batch_size]], pad_token=self.trg_pad_token)) for x in six.moves.range(0, len(order), self.batch_size)]
    return src_ret, trg_ret

class ShuffleBatcher(SentenceBatcher):
  """
  A class to create batches through randomly shuffling without sorting.
  """

  def pack(self, src, trg):
    order = np.random.shuffle(range(len(src)))
    return self.pack_by_order(src, trg, order)

class SortBatcher(SentenceBatcher):
  """
  A template class to create batches through bucketing sent length.
  """

  def __init__(self, batch_size, src_pad_token=None, trg_pad_token=None, sort_key=lambda x: len(x[0])):
    super(SentenceBatcher, self).__init__(batch_size, src_pad_token=src_pad_token, trg_pad_token=trg_pad_token)
    self.sort_key = sort_key

  def pack(self, src, trg):
    order = np.argsort([self.sort_key(x) for x in zip(src,trg)])
    return self.pack_by_order(src, trg, order)
