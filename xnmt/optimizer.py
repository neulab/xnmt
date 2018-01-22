import dynet as dy
import numpy as np
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
"""
The purpose of this module is mostly to expose the DyNet trainers to YAML serialization,
but may also be extended to customize optimizers / training schedules
"""

class XnmtOptimizer(object):
  def update(self): return self.optimizer.update()
  def update_epoch(self, r = 1.0): return self.optimizer.update_epoch(r)
  def status(self): return self.optimizer.status()
  def set_clip_threshold(self, thr): return self.optimizer.set_clip_threshold(thr)
  def get_clip_threshold(self): return self.optimizer.get_clip_threshold()
  def restart(self): return self.optimizer.restart()
  @property
  def learning_rate(self):
      return self.optimizer.learning_rate
  @learning_rate.setter
  def learning_rate(self, value):
      self.optimizer.learning_rate = value

class SimpleSGDTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!SimpleSGDTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), e0 = 0.1):
    self.optimizer = dy.SimpleSGDTrainer(xnmt_global.dynet_param_collection.param_col, 
                                         e0)
class MomentumSGDTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!MomentumSGDTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), e0 = 0.01, mom = 0.9):
    self.optimizer = dy.MomentumSGDTrainer(xnmt_global.dynet_param_collection.param_col, 
                                           e0, mom)

class AdagradTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdagradTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), e0 = 0.1, eps = 1e-20):
    self.optimizer = dy.AdagradTrainer(xnmt_global.dynet_param_collection.param_col, 
                                       e0, eps=eps)

class AdadeltaTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdadeltaTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), eps = 1e-6, rho = 0.95):
    self.optimizer = dy.AdadeltaTrainer(xnmt_global.dynet_param_collection.param_col, 
                                        eps, rho)

class AdamTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdamTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8):
    self.optimizer = dy.AdamTrainer(xnmt_global.dynet_param_collection.param_col, 
                                    alpha, beta_1, beta_2, eps)

class TransformerAdamTrainer(XnmtOptimizer, Serializable):
  """
  Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
  In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay
  """
  yaml_tag = u'!TransformerAdamTrainer'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), alpha=1.0, dim=512, warmup_steps=4000, beta_1=0.9, beta_2=0.98, eps=1e-9):
    self.optimizer = dy.AdamTrainer(xnmt_global.dynet_param_collection.param_col,
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
      print('> Optimizer Logging')
      print('  Steps=%d, learning_rate=%.2e' % (self.steps, self.optimizer.learning_rate))