
import dynet as dy

from xnmt.persistence import Serializable, serializable_init
from xnmt.events import handle_xnmt_event, register_xnmt_handler

class LengthPrior(Serializable):
  yaml_tag = '!LengthPrior'
  @serializable_init
  @register_xnmt_handler
  def __init__(self, prior, weight):
    self.prior = prior
    self.weight = weight

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src = src

  def log_ll(self, sampled_actions):
    print(sampled_actions)


