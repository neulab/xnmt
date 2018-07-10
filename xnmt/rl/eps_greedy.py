
import numpy as np

from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.priors import UniformPrior

class EpsilonGreedy(Serializable):
  yaml_tag = '!EpsilonGreedy'

  @serializable_init
  def __init__(self, eps_prob=0, prior=bare(UniformPrior)):
    self.eps_prob = eps_prob
    self.prior = prior
      
  def is_triggered(self): return np.random.random() <= self.eps_prob
  def get_random_func(self): return self.prior

