
import dynet as dy
import math
from scipy.stats import poisson

from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.events import handle_xnmt_event, register_xnmt_handler
from xnmt.priors import PoissonPrior

class PoissonLengthPrior(Serializable):
  yaml_tag = '!PoissonLengthPrior'

  @serializable_init
  @register_xnmt_handler
  def __init__(self, lmbd=3.3, weight=1):
    self.lmbd = 3.3
    self.weight = weight

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src = src

  def log_ll(self, sample):
    batch = [self.weight * math.log(poisson.pmf(len_sample, mu=src.len_unpadded() / self.lmbd))
             for src, len_sample in zip(self.src, sample)]
    return dy.inputTensor(batch, batched=True)

