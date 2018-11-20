from typing import Sequence
import numbers

import dynet as dy

from xnmt.persistence import serializable_init, Serializable
from xnmt.events import register_xnmt_handler, handle_xnmt_event

class ConfidencePenalty(Serializable):
  """ 
  The confidence penalty.
  part of: https://arxiv.org/pdf/1701.06548.pdf 
  
  Calculate the -entropy for the given (batched policy).
  Entropy is used as an additional loss so that it will penalize a too confident network.
  """
 
  yaml_tag = "!ConfidencePenalty"

  @serializable_init
  @register_xnmt_handler
  def __init__(self, weight: numbers.Real = 1.0):
    self.weight = weight

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.valid_pos = src.mask.get_valid_position() if src.mask is not None else None

  def calc_loss(self, policy: Sequence[dy.Expression]) -> dy.Expression:
    if self.weight < 1e-8:
      return None
    neg_entropy = []
    for i, ll in enumerate(policy):
      if self.valid_pos is not None:
        ll = dy.pick_batch_elems(ll, self.valid_pos[i])
      loss = dy.sum_batches(dy.sum_elems(dy.cmult(dy.exp(ll), ll)))
      neg_entropy.append(dy.sum_batches(loss))
    return self.weight * dy.esum(neg_entropy)

