import dynet as dy
import numpy as np
from collections import defaultdict
from vocab import Vocab
from pprint import pprint

class Batcher:
  '''
  A template class to convert a list of sentences to several batches of
  sentences.
  '''

  PAIR_SRC = 0
  PAIR_TRG = 1

  def __init__(self, batch_size):
    self.batch_size = batch_size

  @staticmethod
  def is_batch_sentence(source):
    return len(source) > 0 and type(source[0]) == list

  @staticmethod
  def is_batch_word(source):
    return type(source) == list


class BucketBatcher(Batcher):

  @staticmethod
  def group_by_len(pairs, indexer):
    buckets = defaultdict(list)
    for pair in pairs:
      buckets[indexer(pair)].append(pair)
    return buckets

  @staticmethod
  def separate_source_target(joint_result):
    source_result = []
    target_result = []
    for batch in joint_result:
      source_result.append([pair[0] for pair in batch])
      target_result.append([pair[1] for pair in batch])
    return source_result, target_result

  def create_bucket_batches(self, bucket_sent_pairs):
    minibatches = []
    num_sent_pairs = len(bucket_sent_pairs)
    num_batches = (num_sent_pairs + self.batch_size - 1) // self.batch_size

    for batch_idx in range(num_batches):
      real_batch_size = min(self.batch_size, num_sent_pairs - batch_idx * self.batch_size)
      one_batch = [bucket_sent_pairs[batch_idx * self.batch_size + i] for i in range(real_batch_size)]
      self.pad(one_batch)
      minibatches.append(one_batch)

    np.random.shuffle(minibatches)
    return minibatches

  def pack_template(self, source, target, indexer, sorter=lambda x: x):
    source_target_pairs = zip(source, target)
    buckets = self.group_by_len(source_target_pairs, indexer)

    result = []
    for same_len_pairs in buckets.values():
      sorter(same_len_pairs)
      result.extend(self.create_bucket_batches(same_len_pairs))
    np.random.shuffle(result)
    # pprint(result)

    return self.separate_source_target(result)

  def pad(self, batch):
    raise NotImplementedError('pad must be implemented in BucketBatcher subclasses')


class SourceBucketBatcher(BucketBatcher):

  def pack(self, source, target):
    indexer = lambda x: len(x[self.PAIR_SRC])
    return self.pack_template(source, target, indexer)

  def pad(self, batch):
    pass


class SourceTargetBucketBatcher(SourceBucketBatcher):

  def pack(self, source, target):
    indexer = lambda x: len(x[self.PAIR_SRC])
    sorter = lambda x: x.sort(key=lambda pair: len(pair[self.PAIR_TRG]))
    return self.pack_template(source, target, indexer, sorter)


class TargetBucketBatcher(BucketBatcher):

  def pack(self, source, target):
    indexer = lambda x: len(x[self.PAIR_TRG])
    return self.pack_template(source, target, indexer)

  def pad(self, batch):
    max_len = max([len(pair[self.PAIR_SRC]) for pair in batch])
    for pair in batch:
      if len(pair[self.PAIR_SRC]) < max_len:
        pair[self.PAIR_SRC].extend([Vocab.ES] * (max_len - len(pair[self.PAIR_SRC])))


class TargetSourceBucketBatcher(TargetBucketBatcher):

  def pack(self, source, target):
    indexer = lambda x: len(x[self.PAIR_TRG])
    sorter = lambda x: x.sort(key=lambda pair: len(pair[self.PAIR_SRC]))
    return self.pack_template(source, target, indexer, sorter)


