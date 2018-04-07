import logging
logger = logging.getLogger('xnmt')

import dynet as dy
import numpy as np
from xnmt.param_collection import ParamManager
from xnmt.serializer import serializable_init, Serializable

"""
The purpose of this module is mostly to expose the DyNet trainers to YAML serialization,
but may also be extended to customize optimizers / training schedules
"""

class XnmtOptimizer(object):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of DyNet trainers but can add extra functionality.
  """
  def update(self):
    """
    Update the parameters.
    """
    return self.optimizer.update()

  def status(self):
    """
    Outputs information about the trainer in the stderr.

    (number of updates since last call, number of clipped gradients, learning rate, etc…)
    """
    return self.optimizer.status()

  def set_clip_threshold(self, thr):
    """
    Set clipping thershold

    To deactivate clipping, set the threshold to be <=0

    Args:
      thr (number): Clipping threshold
    """
    return self.optimizer.set_clip_threshold(thr)

  def get_clip_threshold(self):
    """
    Get clipping threshold

    Returns:
      number: Gradient clipping threshold
    """
    return self.optimizer.get_clip_threshold()

  def restart(self):
    """
    Restarts the optimizer

    Clears all momentum values and assimilate (if applicable)
    """
    return self.optimizer.restart()

  @property
  def learning_rate(self):
      return self.optimizer.learning_rate
  @learning_rate.setter
  def learning_rate(self, value):
      self.optimizer.learning_rate = value

class SimpleSGDTrainer(XnmtOptimizer, Serializable):
  """
  Stochastic gradient descent trainer

  This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.

  Args:
    e0 (number): Initial learning rate
  """
  yaml_tag = '!SimpleSGDTrainer'

  @serializable_init
  def __init__(self, e0 = 0.1):
    self.optimizer = dy.SimpleSGDTrainer(ParamManager.global_collection(), e0)
class MomentumSGDTrainer(XnmtOptimizer, Serializable):
  """
  Stochastic gradient descent with momentum

  This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.

  Args:
    e0 (number): Initial learning rate
    mom (number): Momentum
  """
  yaml_tag = '!MomentumSGDTrainer'

  @serializable_init
  def __init__(self, e0 = 0.01, mom = 0.9):
    self.optimizer = dy.MomentumSGDTrainer(ParamManager.global_collection(), e0, mom)

class AdagradTrainer(XnmtOptimizer, Serializable):
  """
  Adagrad optimizer

  The adagrad algorithm assigns a different learning rate to each parameter.

  Args:
    e0 (number): Initial learning rate
    eps (number): Epsilon parameter to prevent numerical instability
  """
  yaml_tag = '!AdagradTrainer'

  @serializable_init
  def __init__(self, e0 = 0.1, eps = 1e-20):
    self.optimizer = dy.AdagradTrainer(ParamManager.global_collection(), e0, eps=eps)

class AdadeltaTrainer(XnmtOptimizer, Serializable):
  """
  AdaDelta optimizer

  The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.

  Args:
    eps (number): Epsilon parameter to prevent numerical instability
    rho (number): Update parameter for the moving average of updates in the numerator
  """
  yaml_tag = '!AdadeltaTrainer'

  @serializable_init
  def __init__(self, eps = 1e-6, rho = 0.95):
    self.optimizer = dy.AdadeltaTrainer(ParamManager.global_collection(), eps, rho)

class AdamTrainer(XnmtOptimizer, Serializable):
  """
  Adam optimizer

  The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient

  Args:
    alpha (number): Initial learning rate
    beta_1 (number): Moving average parameter for the mean
    beta_2 (number): Moving average parameter for the variance
    eps (number): Epsilon parameter to prevent numerical instability
  """
  yaml_tag = '!AdamTrainer'

  @serializable_init
  def __init__(self, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8):
    self.optimizer = dy.AdamTrainer(ParamManager.global_collection(), alpha, beta_1, beta_2, eps)

class TransformerAdamTrainer(XnmtOptimizer, Serializable):
  """
  Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
  In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay

  Args:
    alpha (float):
    dim (int):
    warmup_steps (int):
    beta_1 (float):
    beta_2 (float):
    eps (float):
  """
  yaml_tag = '!TransformerAdamTrainer'

  @serializable_init
  def __init__(self, alpha=1.0, dim=512, warmup_steps=4000, beta_1=0.9, beta_2=0.98, eps=1e-9):
    self.optimizer = dy.AdamTrainer(ParamManager.global_collection(),
                                    alpha=alpha,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    eps=eps)
    self.dim = dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def update(self):
    self.steps += 1
    decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
    self.optimizer.learning_rate = 1. * decay
    self.optimizer.update()

    if self.steps % 200 == 0:
      logger.info('> Optimizer Logging')
      logger.info('  Steps=%d, learning_rate=%.2e' % (self.steps, self.optimizer.learning_rate))
