from __future__ import division, generators

import dynet as dy
from xnmt.serializer import Serializable


class BaseTrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, update_weights=True):
    """
    Runs training steps in a loop until stopping criterion is reached.
    
    :param update_weights: Whether parameters should be updated
    """
    raise NotImplementedError("")

class TrainingTask(object):
  """
  Base class for a training task.
  """
  def __init__(self, model):
    self.model = model
  
  def should_stop_training(self):
    """
    :returns: True iff training is finished, i.e. training_step(...) should not be called again
    """
    raise NotImplementedError("")
  
  def training_step(self, src, trg):
    """
    Performs forward pass corresponding to a single training step.
    Training logic like switching epochs, reshuffling batches, etc. must be
    handled as well.
    
    :param src: src minibatch
    :param trg: trg minibatch
    :returns: Loss
    """
    raise NotImplementedError("")

  def update_weights(self, loss, trainer, dynet_profiling):
    """
    Standardized way to perform backward pass and parameter updates.
    Can be sidestepped e.g. for custom multitask training logic.
    
    :param loss: Result of self.training_step(...)
    :param trainer: DyNet trainer / xnmt.optimizer object
    :param dynet_profiling: if > 0, print the computation graph 
    """
    if dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()
    trainer.update()
  
  def checkpoint_needed(self):
    raise NotImplementedError()

  def checkpoint(self, out_ext=".dev_hyp", ref_ext=".dev_ref", encoding='utf-8'):
    raise NotImplementedError()

class BaseMultiTrainingTask(BaseTrainingRegimen, TrainingTask):
  def __init__(self, model_file, tasks, stopping_criterion="all", dynet_profiling=0):
    self.model_file = model_file
    self.dynet_profiling = dynet_profiling
    self.stopping_criterion = stopping_criterion
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    for task in tasks[1:]:
      if not task.trainer is tasks[0].trainer:
        raise ValueError("Tasks must reference-share Trainer objects!")
    self.train = None
  def trigger_train_event(self, value):
    if self.train is None:
      self.train = value
      self.tasks[0].model.set_train(value)
    else:
      if value!=self.train:
        self.train = value
        self.tasks[0].model.set_train(value)
        
class JointMultiTrainingTask(BaseMultiTrainingTask, Serializable):
  yaml_tag = u"!JointMultiTrainingTask"
  """
  Multi-task training where gradients are accumulated and weight updates
  are thus performed jointly for each task. The relative weight between
  tasks can be configured by setting each tasks batch size accordingly.
  """
  def __init__(self, model_file, tasks, stopping_criterion="all", dynet_profiling=0):
    """
    :param model_file:
    :param tasks: list of TrainingTask instances
    :param stopping_criterion: stop when "all" tasks signal stopping or when "any" task signals stopping
    :param dynet_profiling: if > 0, print computation graph
    """
    super(self, JointMultiTrainingTask).__init__(model_file=model_file,
                                                 tasks=tasks, 
                                                 stopping_criterion=stopping_criterion,
                                                 dynet_profiling=dynet_profiling)
  def run_training(self, update_weights=True):
    self.trigger_train_event(update_weights)
    while True:
      task_losses = []
      for task in self.tasks:
        src, trg = next(task.next_minibatch()) #TODO: used properly??
        task_losses.append(task.training_step(src, trg))
      if update_weights:
        self.update_weights(sum(task_losses), self.tasks[0].trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      for task in self.tasks:
        if task.checkpoint_needed():
          self.trigger_train_event(False)
          task.checkpoint()
      self.trigger_train_event(update_weights)
      if self.stopping_criterion=="all":
        if all([task.should_stop_training() for task in self.tasks]): break
      elif self.stopping_criterion=="any":
        if any([task.should_stop_training() for task in self.tasks]): break

# TODO: implement SerialMultiTrainingTask