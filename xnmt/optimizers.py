from typing import Optional
import numbers

import dynet as dy
import numpy as np

from xnmt import logger
from xnmt.param_collections import ParamManager
from xnmt.persistence import serializable_init, Serializable
from xnmt import utils

"""
The purpose of this module is mostly to expose the DyNet trainers to YAML serialization,
but may also be extended to customize optimizers / training schedules
"""

class XnmtOptimizer(object):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of DyNet trainers but can add extra functionality.

  Args:
    optimizer: the underlying DyNet optimizer (trainer)
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """

  def __init__(self, optimizer: dy.Trainer, skip_noisy: bool = False) -> None:
    self.optimizer = optimizer
    self.skip_noisy = skip_noisy
    if skip_noisy:
      self.rolling_stats = utils.RollingStatistic()

  def update(self) -> None:
    """
    Update the parameters.
    """
    try:
      if not (self.skip_noisy and self._check_gradients_noisy()):
        self.optimizer.update()
      else:
        logger.info("skipping noisy update")
    except RuntimeError:
      logger.warning("Failed to perform update. Skipping example and clearing gradients.")
      for subcol in ParamManager.param_col.subcols.values():
        for param in subcol.parameters_list():
          param.scale_gradient(0)

  def status(self) -> None:
    """
    Outputs information about the trainer in the stderr.

    (number of updates since last call, number of clipped gradients, learning rate, etc…)
    """
    return self.optimizer.status()

  def set_clip_threshold(self, thr: numbers.Real) -> None:
    """
    Set clipping thershold

    To deactivate clipping, set the threshold to be <=0

    Args:
      thr: Clipping threshold
    """
    return self.optimizer.set_clip_threshold(thr)

  def get_clip_threshold(self) -> numbers.Real:
    """
    Get clipping threshold

    Returns:
      Gradient clipping threshold
    """
    return self.optimizer.get_clip_threshold()

  def restart(self) -> None:
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

  def _check_gradients_noisy(self) -> bool:
    sq_norm = 0
    for subcol in ParamManager.param_col.subcols.values():
      for param in subcol.parameters_list():
        cur_grads = param.grad_as_array()
        sq_norm += np.sum(np.square(cur_grads))
    log_norm = np.log(np.sqrt(sq_norm))
    self.rolling_stats.update(log_norm)
    if self.rolling_stats.average is None: # too few statistics
      return False
    else:
      req_min = self.rolling_stats.average - 4*self.rolling_stats.stddev
      req_max = self.rolling_stats.average + 4*self.rolling_stats.stddev
      return not (req_min < log_norm < req_max)

class SimpleSGDTrainer(XnmtOptimizer, Serializable):
  """
  Stochastic gradient descent trainer

  This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.

  Args:
    e0: Initial learning rate
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!SimpleSGDTrainer'

  @serializable_init
  def __init__(self, e0: numbers.Real = 0.1, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.SimpleSGDTrainer(ParamManager.global_collection(), e0),
                     skip_noisy=skip_noisy)

class MomentumSGDTrainer(XnmtOptimizer, Serializable):
  """
  Stochastic gradient descent with momentum

  This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.

  Args:
    e0: Initial learning rate
    mom: Momentum
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!MomentumSGDTrainer'

  @serializable_init
  def __init__(self, e0: numbers.Real = 0.01, mom: numbers.Real = 0.9, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.MomentumSGDTrainer(ParamManager.global_collection(), e0, mom),
                     skip_noisy=skip_noisy)

class AdagradTrainer(XnmtOptimizer, Serializable):
  """
  Adagrad optimizer

  The adagrad algorithm assigns a different learning rate to each parameter.

  Args:
    e0: Initial learning rate
    eps: Epsilon parameter to prevent numerical instability
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!AdagradTrainer'

  @serializable_init
  def __init__(self, e0: numbers.Real = 0.1, eps: numbers.Real = 1e-20, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdagradTrainer(ParamManager.global_collection(), e0, eps=eps),
                     skip_noisy=skip_noisy)

class AdadeltaTrainer(XnmtOptimizer, Serializable):
  """
  AdaDelta optimizer

  The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.

  Args:
    eps: Epsilon parameter to prevent numerical instability
    rho: Update parameter for the moving average of updates in the numerator
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!AdadeltaTrainer'

  @serializable_init
  def __init__(self, eps: numbers.Real = 1e-6, rho: numbers.Real = 0.95, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdadeltaTrainer(ParamManager.global_collection(), eps, rho),
                     skip_noisy=skip_noisy)

class AdamTrainer(XnmtOptimizer, Serializable):
  """
  Adam optimizer

  The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient

  Args:
    alpha: Initial learning rate
    beta_1: Moving average parameter for the mean
    beta_2: Moving average parameter for the variance
    eps: Epsilon parameter to prevent numerical instability
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!AdamTrainer'

  @serializable_init
  def __init__(self,
               alpha: numbers.Real = 0.001,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.999,
               eps: numbers.Real = 1e-8,
               skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(), alpha, beta_1, beta_2, eps),
                     skip_noisy=skip_noisy)

class NoamTrainer(XnmtOptimizer, Serializable):
  """
  Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
  In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay

  Args:
    alpha:
    dim:
    warmup_steps:
    beta_1:
    beta_2:
    eps:
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  yaml_tag = '!NoamTrainer'

  @serializable_init
  def __init__(self,
               alpha: numbers.Real = 1.0,
               dim: numbers.Integral = 512,
               warmup_steps: Optional[numbers.Integral] = 4000,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.98,
               eps: numbers.Real = 1e-9,
               skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(),
                                    alpha=alpha,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    eps=eps),
                     skip_noisy=skip_noisy)
    self.dim = dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def update(self) -> None:
    self.steps += 1
    if self.warmup_steps:
      decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
    else:
      decay = (self.dim ** (-0.5)) * self.steps ** (-0.5)
    self.optimizer.learning_rate = 1. * decay
    super().update()

    if self.steps % 200 == 0:
      logger.info('> Optimizer Logging')
      logger.info('  Steps=%d, learning_rate=%.2e' % (self.steps, self.optimizer.learning_rate))



class DummyTrainer(XnmtOptimizer, Serializable):
  """
  A dummy trainer that does not perform any parameter updates.
  """
  yaml_tag = "!DummyTrainer"

  @serializable_init
  def __init__(self) -> None:
    pass

  def update(self) -> None:
    pass

  def status(self) -> None:
    pass

  def set_clip_threshold(self, thr) -> None:
    pass

  def get_clip_threshold(self) -> None:
    pass

  def restart(self) -> None:
    pass

  @property
  def learning_rate(self):
    return 1.0
  @learning_rate.setter
  def learning_rate(self, value):
    pass
