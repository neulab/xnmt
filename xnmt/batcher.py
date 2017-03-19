import dynet as dy
import numpy as np
from collections import defaultdict


class Batcher:
  '''
  A template class to convert a list of sentences to several batches of
  sentences.
  '''

  @staticmethod
  def is_batch_sentence(source):
    return len(source) > 0 and type(source[0]) == list

  @staticmethod
  def is_batch_word(source):
    return type(source) == list


class SourceBucketBatcher(Batcher):

  def __init__(self, batch_size):
    self.batch_size = batch_size

  def pack(self, source, target):
    source_target_pairs = zip(source, target)
    buckets = self.group_by_source_len(source_target_pairs)

    result = []
    for same_len_src_pairs in buckets.values():
      result.extend(self.create_minibatches(same_len_src_pairs))
    np.random.shuffle(result)

    source_result = []
    target_result = []
    for batch in result:
      source_result.append([pair[0] for pair in batch])
      target_result.append([pair[1] for pair in batch])

    return source_result, target_result

  def group_by_source_len(self, pairs):
    buckets = defaultdict(list)
    for pair in pairs:
      buckets[len(pair[0])].append(pair)

    return buckets

  def create_minibatches(self, pairs):
    minibatches = []
    num_pairs = len(pairs)
    num_batches = (num_pairs + self.batch_size - 1) // self.batch_size
    for batch_idx in range(num_batches):
      real_batch_size = min(self.batch_size, num_pairs - batch_idx * self.batch_size)
      minibatches.append([pairs[batch_idx * self.batch_size + i] for i in range(real_batch_size)])
    np.random.shuffle(minibatches)

    return minibatches
