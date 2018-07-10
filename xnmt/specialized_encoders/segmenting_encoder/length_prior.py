
from xnmt.persistence import Serializable, serializable_init

class LengthPrior(Serializable):
  yaml_tag = '!LengthPrior'
  @serializable_init
  def __init__(self, prior, weight):
    self.prior = prior
    self.weight = weight

  def calculate_ll(self, sampled_actions):
    pass  
               ## OPTIONS


