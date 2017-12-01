import dynet as dy
from xnmt.serializer import Serializable

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
  def __init__(self, yaml_context, e0 = 0.1):
    self.optimizer = dy.SimpleSGDTrainer(yaml_context.dynet_param_collection.param_col, 
                                         e0)
class MomentumSGDTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!MomentumSGDTrainer'
  def __init__(self, yaml_context, e0 = 0.01, mom = 0.9):
    self.optimizer = dy.MomentumSGDTrainer(yaml_context.dynet_param_collection.param_col, 
                                           e0, mom)

class AdagradTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdagradTrainer'
  def __init__(self, yaml_context, e0 = 0.1, eps = 1e-20):
    self.optimizer = dy.AdagradTrainer(yaml_context.dynet_param_collection.param_col, 
                                       e0, eps=eps)

class AdadeltaTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdadeltaTrainer'
  def __init__(self, yaml_context, eps = 1e-6, rho = 0.95):
    self.optimizer = dy.AdadeltaTrainer(yaml_context.dynet_param_collection.param_col, 
                                        eps, rho)

class AdamTrainer(XnmtOptimizer, Serializable):
  yaml_tag = u'!AdamTrainer'
  def __init__(self, yaml_context, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8):
    self.optimizer = dy.AdamTrainer(yaml_context.dynet_param_collection.param_col, 
                                    alpha, beta_1, beta_2, eps)
