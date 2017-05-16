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
  A template class to convert a list of sentences to several batches of sentences.
  """

  PAIR_SRC = 0
  PAIR_TRG = 1

  def __init__(self, batch_size, pad_token=Vocab.ES):
    self.batch_size = batch_size
    self.pad_token = pad_token

  @staticmethod
  def is_batch_sentence(sentence):
    """
    :rtype: bool
    :param sentence: a batch of sentences (a list of lists of ints) OR a single sentence (a list of ints)
    :return: True if the data is a batch of sentences 
    """
    return type(sentence) == Batch

  @staticmethod
  def is_batch_word(word):
    """
    :rtype: bool
    :param word: a batch of target words (a list of ints) OR a single target word (an int)
    :return: True if the data is a batch of target words 
    """
    return type(word) == Batch

  @staticmethod
  def mark_as_batch(data):
    if type(data) == Batch:
      return data
    else:
      return Batch(data)

  @staticmethod
  def separate_source_target(joint_result):
    source_result = []
    target_result = []
    for batch in joint_result:
      source_result.append(Batcher.mark_as_batch([pair[Batcher.PAIR_SRC] for pair in batch]))
      target_result.append(Batcher.mark_as_batch([pair[Batcher.PAIR_TRG] for pair in batch]))
    return source_result, target_result

  @staticmethod
  def pad_src_sent(batch, pad_token=Vocab.ES):
    max_len = max([len(pair[Batcher.PAIR_SRC]) for pair in batch])
    return map(lambda pair:
               (pair[Batcher.PAIR_SRC].get_padded_sentence(pad_token, max_len - len(pair[Batcher.PAIR_SRC])),
                pair[Batcher.PAIR_TRG]),
               batch)

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
    return self.pad_src_sent(batch, self.pad_token)

  def pack(self, source, target):
    """
    Create batches from input source and target corpus.
    :param source: source corpus (a list of sentences)
    :param target: target corpus (a list of sentences)
    :return: Packed source corpus (a list of batches of sentence) and packed target corpus (a list of batches of 
    sentence)
    """
    raise NotImplementedError('pack() must be implemented in Batcher subclasses')


class ShuffleBatcher(Batcher):
  """
  A class to create batches through randomly shuffling without sorting.
  """

  def pack(self, source, target):
    source_target_pairs = list(zip(source, target))
    np.random.shuffle(source_target_pairs)
    minibatches = self.create_batches(source_target_pairs)
    return self.separate_source_target(minibatches)

  def pad_sent(self, batch):
    return self.pad_src_sent(batch, self.pad_token)


class BucketBatcher(Batcher):
  """
  A template class to create batches through bucketing sentence length.
  """

  def group_by_len(self, pairs):
    buckets = defaultdict(list)
    for pair in pairs:
      buckets[self.bucket_index(pair)].append(pair)
    return buckets

  def pack(self, source, target):
    source_target_pairs = zip(source, target)
    buckets = self.group_by_len(source_target_pairs)
    sorted_pairs = []
    for bucket_key in sorted(buckets.keys()):
      same_len_pairs = buckets[bucket_key]
      self.bucket_value_sort(same_len_pairs)
      sorted_pairs.extend(same_len_pairs)
    result = self.create_batches(sorted_pairs)
    np.random.shuffle(result)
    return self.separate_source_target(result)

  def bucket_index(self, pair):
    """
    Specify the method to sort sentences.
    """
    raise NotImplementedError('bucket_index() must be implemented in BucketBatcher subclasses')

  def bucket_value_sort(self, pairs):
    """
    Specify the method to break ties for sorted sentences.
    """
    np.random.shuffle(pairs)


class SourceBucketBatcher(BucketBatcher):
  """
  A class to create batches based on the source sentence length.
  """

  def bucket_index(self, pair):
    return len(pair[Batcher.PAIR_SRC])


class SourceTargetBucketBatcher(SourceBucketBatcher):
  """
  A class to create batches based on the source sentence length and break ties by target sentence length.
  """

  def bucket_value_sort(self, pairs):
    return pairs.sort(key=lambda pair: len(pair[Batcher.PAIR_TRG]))


class TargetBucketBatcher(BucketBatcher):
  """
  A class to create batches based on the target sentence length.
  """

  def bucket_index(self, pair):
    return len(pair[Batcher.PAIR_TRG])

  def pad_sent(self, batch):
    return self.pad_src_sent(batch, self.pad_token)


class TargetSourceBucketBatcher(TargetBucketBatcher):
  """
  A class to create batches based on the target sentence length and break ties by source sentence length.
  """

  def bucket_value_sort(self, pairs):
    return pairs.sort(key=lambda pair: len(pair[Batcher.PAIR_SRC]))


class WordTargetBucketBatcher(TargetBucketBatcher):
  """
  A class to create batches based on number of target words, resulting in more stable memory consumption.
  """

  def pack(self, source, target):
    limit_target_words = self.batch_size
    source_target_pairs = zip(source, target)
    buckets = self.group_by_len(source_target_pairs)

    result = []
    temp_batch = []
    temp_words = 0

    for sent_len, sent_pairs in OrderedDict(buckets).items():
      self.bucket_value_sort(sent_pairs)
      for pair in sent_pairs:
        if temp_words + sent_len > limit_target_words and len(temp_batch) > 0:
          result.append(self.pad_sent(temp_batch))
          temp_batch = []
          temp_words = 0
        temp_batch.append(pair)
        temp_words += sent_len

    if temp_words != 0:
      result.append(self.pad_sent(temp_batch))

    np.random.shuffle(result)
    return self.separate_source_target(result)
