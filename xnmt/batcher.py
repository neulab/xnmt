import dynet as dy
import numpy as np
from collections import defaultdict

class Batcher:

  def __init__(self, minibatch_size):
    self.minibatch_size = minibatch_size

  def pack(self, source):
    buckets = self.group_by_len(source)
    result = []
    for sentences in buckets.values():
      result.extend(self.create_minibatches(sentences))
    np.random.shuffle(result)
    return result

  def group_by_len(self, sentences):
    buckets = defaultdict(list)
    for sentence in sentences:
      buckets[len(sentence)].append(sentence)
    return buckets

  def create_minibatches(self, sentences):
    num_sentences = len(sentences)
    minibatches = []
    for i in range(0, num_sentences - 1, self.minibatch_size):
      real_batch_size = self.minibatch_size if i + self.minibatch_size <= num_sentences else num_sentences - i
      minibatches.append([sentences[i + ii] for ii in range(real_batch_size)])
    return minibatches

  @staticmethod
  def is_batch_sentence(source):
    return len(source) > 0 and type(source[0]) == list

  @staticmethod
  def is_batch_word(source):
    return type(source) == list
