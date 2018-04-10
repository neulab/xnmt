import dynet as dy
import numpy as np

from xnmt import logger
from xnmt.persistence import Serializable, Ref, Path

class NegativeLogProbLossScaler(Serializable):
  yaml_tag = '!LossCalculator'
  
  def __init__(self, trg_vocab=Ref(Path("model.trg_reader.vocab")), word_count=None):
    counts = {}
    total_count = 0
    with open(word_count) as fp:
      for line in word_count:
        line = line.strip().split()
        word = " ".join(line[:-1])
        count = int(line[-1])
        if word in trg_vocab:
          counts[word] = count
          total_count += count
        else:
          logger.debug("WARNING, ignoring because it is not contained in vocab:", word)
     




  def __call__(self, loss_builder):
    mle_loss = loss_builder["mle"]

    if mle_loss:
      pass
  

