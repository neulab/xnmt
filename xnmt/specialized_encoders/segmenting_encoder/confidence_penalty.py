""" 
  https://arxiv.org/pdf/1701.06548.pdf 
"""

from xnmt.persistence import serializable_init, Serializable

class SegmentationConfidencePenalty(Serializable):
  
  yaml_tag = "!SegmentationConfidencePenalty"

  """
  strength: the beta value
  """
  @serializable_init
  def __init__(self, strength):
    self.strength = strength
    if strength.value() < 0:
      raise RuntimeError("Strength of label smoothing parameter should be >= 0")

  def penalty(self, logsoftmaxes, mask):
    strength = self.strength.value()
    if strength < 1e-8:
      return None
    neg_entropy = []
    for i, logsoftmax in enumerate(logsoftmaxes):
      loss = dy.cmult(dy.exp(logsoftmax), logsoftmax)
      if mask is not None:
        loss = dy.cmult(dy.inputTensor(mask[i], batched=True), loss)
      neg_entropy.append(loss)

    return strength * dy.sum_elems(dy.esum(neg_entropy))

  def value(self):
    return self.strength.value()

  def __str__(self):
    return str(self.strength.value())

  def __repr__(self):
    return repr(self.strength.value())

