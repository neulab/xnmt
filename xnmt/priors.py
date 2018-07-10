from scipy.stats import poisson

from xnmt.batcher import is_batched
from xnmt.persistence import serializable_init, Serializable

class PoissonPrior(Serializable):
  yaml_tag = '!PoissonPrior'
  @serializable_init
  def __init__(self, mu=3.3):
    self.mu = mu

  def log_likelihood(self, actual, expected):
    if is_batched(actual):
      assert len(actual) == len(expected)
      lls = [poisson.pmf(act, exp) for act, exp in zip(actual, expected)]
      return dy.inputTensor(lls, batched=True)
    else:
      return dy.scalarInput(poisson.pmf(actual, expected))

class UniformPrior(Serializable):
  yaml_tag = '!UniformPrior'
  @serializable_init
  def __init__(self, low=0, high=1):
    self.x_diff = high - low



