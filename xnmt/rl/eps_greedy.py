"""
Epsilon greedy is a simple component that tells you if you should sample from your policy or from its prior.
When eps_prob = 0, you will always sample from the policy.
When eps_prob = 1, you will always sample from the prior.
"""
import numbers

import numpy as np

from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.specialized_encoders.segmenting_encoder import priors

class EpsilonGreedy(Serializable):
  """
  Args:
    eps_prob: The probability of sampling from the prior.
  """
  yaml_tag = '!EpsilonGreedy'

  @serializable_init
  def __init__(self, eps_prob: numbers.Real = 0.0, prior: priors.Prior = bare(priors.UniformPrior)) -> None:
    self.eps_prob = eps_prob
    self.prior = prior

  def is_triggered(self) -> bool: return np.random.random() <= self.eps_prob
  def get_prior(self) -> priors.Prior: return self.prior

