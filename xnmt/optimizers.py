from typing import Optional
import numbers
import collections

import numpy as np

import xnmt
import xnmt.tensor_tools as tt
from xnmt.settings import settings
from xnmt import logger, tee
from xnmt.param_collections import ParamManager
from xnmt.persistence import serializable_init, Serializable
from xnmt import utils

if xnmt.backend_torch:
  import torch.optim
if xnmt.backend_dynet:
  import dynet as dy

"""
The purpose of this module is mostly to expose the DyNet trainers to YAML serialization,
but may also be extended to customize optimizers / training schedules
"""


class XnmtOptimizer(object):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of the backend trainers but can add extra
  functionality.
  """
  def __init__(self):
    self.learning_rate = None

  def update(self) -> None:
    """
    Update the parameters.
    """
    raise NotImplementedError()

  def restart(self) -> None:
    """
    Restarts the optimizer

    Clears all momentum values and assimilate (if applicable)
    """
    raise NotImplementedError()

@xnmt.require_dynet
class XnmtOptimizerDynet(XnmtOptimizer):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of DyNet trainers but can add extra functionality.

  Args:
    optimizer: the underlying DyNet optimizer (trainer)
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """

  def __init__(self, optimizer: 'dy.Trainer', skip_noisy: bool = False, clip_grads: numbers.Real = 5.0) -> None:
    self.optimizer = optimizer
    self.optimizer.set_clip_threshold(clip_grads)
    self.skip_noisy = skip_noisy
    if skip_noisy:
      self.rolling_stats = utils.RollingStatistic()
    self.global_step = 0

  def update(self) -> None:
    """
    Update the parameters.
    """
    self.global_step += 1
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
    if settings.USE_TENSORBOARD: tee.tensorboard_writer.add_scalars(name="grad", tag_scalar_dict={"norm": np.exp(log_norm)}, global_step=self.global_step)
    self.rolling_stats.update(log_norm)
    if self.rolling_stats.average is None: # too few statistics
      return False
    else:
      req_min = self.rolling_stats.average - 4*self.rolling_stats.stddev
      req_max = self.rolling_stats.average + 4*self.rolling_stats.stddev
      return not (req_min < log_norm < req_max)

@xnmt.require_torch
class XnmtOptimizerTorch(XnmtOptimizer):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of PyTorch trainers but can add extra functionality.

  Args:
    optimizer: the underlying PyTorch optimizer
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
    rescale_grads: rescale gradients if the observed norm should be larger than this given norm
  """

  def __init__(self,
               optimizer: 'torch.optim.Optimizer',
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 0.0,
               rescale_grads: numbers.Real = 15.0) -> None:
    self.optimizer = optimizer
    self.clip_grads = clip_grads
    self.rescale_grads = rescale_grads
    if self.clip_grads > 0.0 and self.rescale_grads > 0.0:
      raise ValueError("either rescale_grads or clip_grads must be deactivated")
    self.lr_factor = 1.0
    self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                       lr_lambda = lambda epoch: self.lr_factor, last_epoch=-1)
    self.skip_noisy = skip_noisy
    if skip_noisy:
      self.rolling_stats = utils.RollingStatistic()
    self.global_step = 0

  def update(self) -> None:
    self.global_step += 1
    if self.rescale_grads > 0.0:
      torch.nn.utils.clip_grad_norm_(ParamManager.global_collection().parameters(), self.rescale_grads)
    elif self.clip_grads > 0.0:
      torch.nn.utils.clip_grad_value_(ParamManager.global_collection().parameters(), self.clip_grads)
    self.scheduler.step()
    if not (self.skip_noisy and self._check_gradients_noisy()):
      self.optimizer.step()
    else:
      logger.info("skipping noisy update")


  def restart(self) -> None:
    # https://discuss.pytorch.org/t/reset-adaptive-optimizer-state/14654/3
    self.optimizer.state = collections.defaultdict(dict)

  @property
  def learning_rate(self):
    return self.lr_factor
  @learning_rate.setter
  def learning_rate(self, value):
    self.lr_factor = value

  def _check_gradients_noisy(self) -> bool:
    sq_norm = 0
    for subcol in ParamManager.param_col.subcols.values():
      for _, param in subcol.named_parameters():
        if param.grad is not None:
          cur_grads = tt.npvalue(param.grad)
          sq_norm += np.sum(np.square(cur_grads))
    log_norm = np.log(np.sqrt(sq_norm))
    if settings.USE_TENSORBOARD: tee.tensorboard_writer.add_scalars(name="grad", tag_scalar_dict={"norm": np.exp(log_norm)}, global_step=self.global_step)
    self.rolling_stats.update(log_norm)
    if self.rolling_stats.average is None: # too few statistics
      return False
    else:
      req_min = self.rolling_stats.average - 4*self.rolling_stats.stddev
      req_max = self.rolling_stats.average + 4*self.rolling_stats.stddev
      return not (req_min < log_norm < req_max)

@xnmt.require_dynet
class SimpleSGDTrainerDynet(XnmtOptimizerDynet, Serializable):
  """
  Stochastic gradient descent trainer

  This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.

  Args:
    e0: Initial learning rate
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """
  yaml_tag = '!SimpleSGDTrainer'

  @serializable_init
  def __init__(self, e0: numbers.Real = 0.1, skip_noisy: bool = False, clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.SimpleSGDTrainer(ParamManager.global_collection(), e0),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads)

@xnmt.require_torch
class SimpleSGDTrainerTorch(XnmtOptimizerTorch, Serializable):
  """
  Stochastic gradient descent trainer

  This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.

  Args:
    e0: Initial learning rate
    momentum: momentum factor
    weight_decay: weight decay (L2 penalty)
    dampening: dampening for momentum
    nesterov: enables Nesterov momentum
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
    rescale_grads: rescale gradients if the observed norm should be larger than this given norm
  """
  yaml_tag = '!SimpleSGDTrainer'

  @serializable_init
  def __init__(self,
               e0: numbers.Real = 0.1,
               momentum: numbers.Real = 0.0,
               weight_decay: numbers.Real = 0.0,
               dampening: numbers.Real = 0.0,
               nesterov: bool = False,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 0.0,
               rescale_grads: numbers.Real = 15.0) -> None:
    super().__init__(optimizer=torch.optim.SGD(params=ParamManager.global_collection().parameters(),
                                               lr=e0,
                                               momentum=momentum,
                                               weight_decay=weight_decay,
                                               dampening=dampening,
                                               nesterov=nesterov),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads,
                     rescale_grads=rescale_grads)

SimpleSGDTrainer = xnmt.resolve_backend(SimpleSGDTrainerDynet, SimpleSGDTrainerTorch)

@xnmt.require_dynet
class MomentumSGDTrainer(XnmtOptimizerDynet, Serializable):
  """
  Stochastic gradient descent with momentum

  This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.

  Args:
    e0: Initial learning rate
    mom: Momentum
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """
  yaml_tag = '!MomentumSGDTrainer'

  @serializable_init
  def __init__(self,
               e0: numbers.Real = 0.01,
               mom: numbers.Real = 0.9,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.MomentumSGDTrainer(ParamManager.global_collection(), e0, mom),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads)

@xnmt.require_dynet
class AdagradTrainer(XnmtOptimizerDynet, Serializable):
  """
  Adagrad optimizer

  The adagrad algorithm assigns a different learning rate to each parameter.

  Args:
    e0: Initial learning rate
    eps: Epsilon parameter to prevent numerical instability
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """
  yaml_tag = '!AdagradTrainer'

  @serializable_init
  def __init__(self,
               e0: numbers.Real = 0.1,
               eps: numbers.Real = 1e-20,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.AdagradTrainer(ParamManager.global_collection(), e0, eps=eps),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads)

@xnmt.require_dynet
class AdadeltaTrainer(XnmtOptimizerDynet, Serializable):
  """
  AdaDelta optimizer

  The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.

  Args:
    eps: Epsilon parameter to prevent numerical instability
    rho: Update parameter for the moving average of updates in the numerator
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """
  yaml_tag = '!AdadeltaTrainer'

  @serializable_init
  def __init__(self,
               eps: numbers.Real = 1e-6,
               rho: numbers.Real = 0.95,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.AdadeltaTrainer(ParamManager.global_collection(), eps, rho),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads)

@xnmt.require_dynet
class AdamTrainerDynet(XnmtOptimizerDynet, Serializable):
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
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
  """
  yaml_tag = '!AdamTrainer'

  @serializable_init
  def __init__(self,
               alpha: numbers.Real = 0.001,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.999,
               eps: numbers.Real = 1e-8,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(), alpha, beta_1, beta_2, eps),
                     skip_noisy=skip_noisy, clip_grads=clip_grads)

@xnmt.require_torch
class AdamTrainerTorch(XnmtOptimizerTorch, Serializable):
  """
  Adam optimizer

  The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient

  Args:
    alpha: Initial learning rate
    beta_1: Moving average parameter for the mean
    beta_2: Moving average parameter for the variance
    eps: Epsilon parameter to prevent numerical instability
    weight_decay: weight decay (L2 penalty)
    amsgrad: whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
    rescale_grads: rescale gradients if the observed norm should be larger than this given norm
  """
  yaml_tag = '!AdamTrainer'

  @serializable_init
  def __init__(self,
               alpha: numbers.Real = 0.001,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.999,
               eps: numbers.Real = 1e-8,
               weight_decay: numbers.Real = 0.0,
               amsgrad: bool = False,
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 0.0,
               rescale_grads: numbers.Real = 15.0) -> None:
    super().__init__(optimizer=torch.optim.Adam(params=ParamManager.global_collection().parameters(),
                                                lr=alpha,
                                                betas=(beta_1, beta_2),
                                                eps=eps,
                                                weight_decay=weight_decay,
                                                amsgrad=amsgrad),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads,
                     rescale_grads=rescale_grads,
                     )

AdamTrainer = xnmt.resolve_backend(AdamTrainerDynet, AdamTrainerTorch)


@xnmt.require_dynet
class NoamTrainerDynet(XnmtOptimizerDynet, Serializable):
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
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
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
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 5.0) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(),
                                    alpha=alpha,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    eps=eps),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads)
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
      logger.info(f'  Steps={self.steps}, learning_rate={self.optimizer.learning_rate:.2e}')

@xnmt.require_torch
class NoamTrainerTorch(XnmtOptimizerTorch, Serializable):
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
    clip_grads: threshold for gradient clipping (to deactivate clipping, set the threshold to be <=0)
    rescale_grads: rescale gradients if the observed norm should be larger than this given norm
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
               skip_noisy: bool = False,
               clip_grads: numbers.Real = 0.0,
               rescale_grads: numbers.Real = 15.0) -> None:
    super().__init__(optimizer=torch.optim.Adam(params=ParamManager.global_collection().parameters(),
                                                lr=alpha, betas=(beta_1, beta_2), eps=eps),
                     skip_noisy=skip_noisy,
                     clip_grads=clip_grads,
                     rescale_grads=rescale_grads)
    self.dim = dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def update(self) -> None:
    self.steps += 1
    if self.warmup_steps:
      decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
    else:
      decay = (self.dim ** (-0.5)) * self.steps ** (-0.5)
    self.lr_factor = 1. * decay
    super().update()

    if self.steps % 200 == 0:
      logger.info('> Optimizer Logging')
      logger.info(f'  Steps={self.steps}, learning_rate={self.lr_factor:.2e}')

NoamTrainer = xnmt.resolve_backend(NoamTrainerDynet, NoamTrainerTorch)


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

  def restart(self) -> None:
    pass

  @property
  def learning_rate(self):
    return 1.0
  @learning_rate.setter
  def learning_rate(self, value):
    pass
