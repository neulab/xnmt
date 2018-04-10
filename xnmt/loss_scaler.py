import dynet as dy
import numpy as np

import xnmt.batcher

from xnmt import logger
from xnmt.events import handle_xnmt_event, register_xnmt_handler
from xnmt.persistence import serializable_init, Serializable, Ref, Path

class NegativeLogProbLossScaler(Serializable):
  yaml_tag = '!NegativeLogProbLossScaler'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, trg_vocab=Ref(Path("model.trg_reader.vocab")), word_count=None, expected_vocab_size=1e6):
    self.weight = np.zeros(len(trg_vocab))
    total_count = 0
    with open(word_count) as fp:
      for line in fp:
        line = line.strip().split()
        word = " ".join(line[:-1])
        count = int(line[-1])
        if word in trg_vocab:
          x = trg_vocab.convert(word)
          self.weight[trg_vocab.convert(word)] = count
          total_count += count
        else:
          logger.debug("WARNING, ignoring because it is not contained in vocab: " + word)
    
    unigram_prob = 0.95 * (self.weight / total_count) + (0.05 / expected_vocab_size)
    self.weight = -np.log(unigram_prob)
    self.first = True
    self.trg_vocab = trg_vocab
    self.train = False
    
  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def __call__(self, loss, ref_action):
    if not self.train:
      return loss

    # Weight the unknown word + eos to be a default weight
    if self.first:
      self.first = False
      self.weight[self.trg_vocab.SS] = 1
      self.weight[self.trg_vocab.ES] = 1
      if hasattr(self.trg_vocab, "unk_token"):
        self.weight[self.trg_vocab.unk_token] = 1

    weight = dy.nobackprop(dy.pick_batch(dy.inputTensor(self.weight), ref_action))
    return dy.cmult(weight, loss)
