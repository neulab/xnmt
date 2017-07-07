from __future__ import division, generators

import numpy as np
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

  PAIR_SRC = 0
  PAIR_TRG = 1

  def __init__(self, batch_size, pad_token=None):
    self.batch_size = batch_size
    # The only reason why we don't set Vocab.ES as the default is because it currently
    # breaks our documentation pipeline
    self.pad_token = pad_token if pad_token != None else Vocab.ES

  @staticmethod
  def is_batch_sent(sent):
    """
    :rtype: bool
    :param sent: a batch of sents (a list of lists of ints) OR a single sent (a list of ints)
    :return: True if the data is a batch of sents
    """
    return type(sent) == Batch

  @staticmethod
  def is_batch_word(word):
    """
    :rtype: bool
    :param word: a batch of trg words (a list of ints) OR a single trg word (an int)
    :return: True if the data is a batch of trg words
    """
    return type(word) == Batch

  @staticmethod
  def mark_as_batch(data):
    if type(data) == Batch:
      return data
    else:
      return Batch(data)

  @staticmethod
  def separate_src_trg(joint_result):
    src_result = []
    trg_result = []
    for batch in joint_result:
      src_result.append(Batcher.mark_as_batch([pair[Batcher.PAIR_SRC] for pair in batch]))
      trg_result.append(Batcher.mark_as_batch([pair[Batcher.PAIR_TRG] for pair in batch]))
    return src_result, trg_result

  @staticmethod
  def pad(batch, pad_token=Vocab.ES):
    # Determine the type of batch
    if len(batch) == 0:
      return batch

    first = batch[0]
    # Case when we are dealing batch of [[f_1, f_2, ..., f_n], [e_1, e_2, ..., e_m]]
    if isinstance(first, (list, tuple)):
      max_len = max(len(src) for src, trg in batch)
      return [(src.get_padded_sent(pad_token, max_len - len(src)), trg) for src, trg in  batch]
    # Case of [[w_1, w_2, ..., w_n]]
    else:
      max_len = max(len(item) for item in batch)
      return [item.get_padded_sent(pad_token, max_len - len(item)) for item in batch]

  @staticmethod
  def select_batcher(batcher_str):
    if batcher_str == 'src':
      return SourceBucketBatcher
    elif batcher_str == 'trg':
      return TargetBucketBatcher
    elif batcher_str == 'src_trg':
      return SourceTargetBucketBatcher
    elif batcher_str == 'trg_src':
      return TargetSourceBucketBatcher
    elif batcher_str == 'shuffle':
      return ShuffleBatcher,
    elif batcher_str == 'word':
      return WordTargetBucketBatcher

  def create_batches(self, sent_pairs):
    minibatches = []
    for batch_start in range(0, len(sent_pairs), self.batch_size):
      one_batch = sent_pairs[batch_start:batch_start+self.batch_size]
      minibatches.append(Batcher.mark_as_batch(self.pad_sent(one_batch)))
    return minibatches

  def pad_sent(self, batch):
    return Batcher.pad(batch, self.pad_token)

  def pack(self, src, trg):
    """
    Create batches from input src and trg corpus.
    :param src: src corpus (a list of sents)
    :param trg: trg corpus (a list of sents)
    :return: Packed src corpus (a list of batches of sent) and packed trg corpus (a list of batches of 
    sent)
    """
    raise NotImplementedError('pack() must be implemented in Batcher subclasses')


class ShuffleBatcher(Batcher):
  """
  A class to create batches through randomly shuffling without sorting.
  """

  def pack(self, src, trg):
    src_trg_pairs = list(zip(src, trg))
    np.random.shuffle(src_trg_pairs)
    minibatches = self.create_batches(src_trg_pairs)
    return self.separate_src_trg(minibatches)

class BucketBatcher(Batcher):
  """
  A template class to create batches through bucketing sent length.
  """

  def group_by_len(self, pairs):
    buckets = defaultdict(list)
    for pair in pairs:
      buckets[self.bucket_index(pair)].append(pair)
    return buckets

  def pack(self, src, trg):
    src_trg_pairs = zip(src, trg)
    buckets = self.group_by_len(src_trg_pairs)
    sorted_pairs = []
    for bucket_key in sorted(buckets.keys()):
      same_len_pairs = buckets[bucket_key]
      self.bucket_value_sort(same_len_pairs)
      sorted_pairs.extend(same_len_pairs)
    result = self.create_batches(sorted_pairs)
    np.random.shuffle(result)
    return self.separate_src_trg(result)

  def bucket_index(self, pair):
    """
    Specify the method to sort sents.
    """
    raise NotImplementedError('bucket_index() must be implemented in BucketBatcher subclasses')

  def bucket_value_sort(self, pairs):
    """
    Specify the method to break ties for sorted sents.
    """
    np.random.shuffle(pairs)


class SourceBucketBatcher(BucketBatcher):
  """
  A class to create batches based on the src sent length.
  """

  def bucket_index(self, pair):
    return len(pair[Batcher.PAIR_SRC])


class SourceTargetBucketBatcher(SourceBucketBatcher):
  """
  A class to create batches based on the src sent length and break ties by trg sent length.
  """

  def bucket_value_sort(self, pairs):
    return pairs.sort(key=lambda pair: len(pair[Batcher.PAIR_TRG]))


class TargetBucketBatcher(BucketBatcher):
  """
  A class to create batches based on the trg sent length.
  """

  def bucket_index(self, pair):
    return len(pair[Batcher.PAIR_TRG])


class TargetSourceBucketBatcher(TargetBucketBatcher):
  """
  A class to create batches based on the trg sent length and break ties by src sent length.
  """

  def bucket_value_sort(self, pairs):
    return pairs.sort(key=lambda pair: len(pair[Batcher.PAIR_SRC]))


class WordTargetBucketBatcher(TargetBucketBatcher):
  """
  A class to create batches based on number of trg words, resulting in more stable memory consumption.
  """

  def pack(self, src, trg):
    limit_trg_words = self.batch_size
    src_trg_pairs = zip(src, trg)
    buckets = self.group_by_len(src_trg_pairs)

    result = []
    temp_batch = []
    temp_words = 0

    for sent_len, sent_pairs in OrderedDict(buckets).items():
      self.bucket_value_sort(sent_pairs)
      for pair in sent_pairs:
        if temp_words + sent_len > limit_trg_words and len(temp_batch) > 0:
          result.append(self.pad_sent(temp_batch))
          temp_batch = []
          temp_words = 0
        temp_batch.append(pair)
        temp_words += sent_len

    if temp_words != 0:
      result.append(self.pad_sent(temp_batch))

    np.random.shuffle(result)
    print("WordTargetBucketBatcher avg batch size: %s sents" % (float(sum([len(x) for x in result]))/len(result)))
    return self.separate_src_trg(result)
