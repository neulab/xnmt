
import math
import numpy as np
from scipy.stats import poisson

from xnmt.persistence import serializable_init, Serializable
from xnmt.events import register_xnmt_handler, handle_xnmt_event

class Prior(object):
  def log_ll(self, event): raise NotImplementedError()
  def sample(self, size): raise NotImplementedError()

class PoissonPrior(Serializable):
  yaml_tag = '!PoissonPrior'
  @serializable_init
  def __init__(self, mu=3.3):
    self.mu = mu

  def log_ll(self, event):
    return poisson.pmf(event, self.mu)

  def sample(self, batch_size, size):
    return np.random.poisson(lam=self.mu, size=(batch_size, size))

class UniformPrior(Serializable):
  yaml_tag = '!UniformPrior'
  @serializable_init
  def __init__(self, low=0, high=1):
    self.x_diff = high - low

  def log_ll(self, event):
    return -math.log(self.x_diff)

  def sample(self, batch_size, size):
    return np.random.uniform(0, self.x_diff, size=(batch_size, size))

class GoldInputPrior(Serializable):
  yaml_tag = '!GoldInputPrior'

  @serializable_init
  @register_xnmt_handler
  def __init__(self, attr_name):
    self.attr_name = attr_name

  def log_ll(self, event):
    return 0

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src = src

  def sample(self, batch_size, size):
    return [getattr(self.src[i], self.attr_name) for i in range(len(batch_size))]

