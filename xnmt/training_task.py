from __future__ import division, generators

from collections import OrderedDict
import numpy as np
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
  Base class for a training task. Training tasks can perform training steps
  and keep track of the training state, but may not implement the actual training
  loop.
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
    if dynet_profiling and dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()
    trainer.update()
  
  def checkpoint_needed(self):
    raise NotImplementedError()

  def checkpoint(self, out_ext=".dev_hyp", ref_ext=".dev_ref", encoding='utf-8'):
    raise NotImplementedError()

class BaseMultiTrainingTask(BaseTrainingRegimen, TrainingTask):
  """
  Base class for multi-task training classes.
  Mainly initializes tasks, performs sanity-checks, and manages set_train events.
  """
  def __init__(self, tasks, stopping_criterion=0, dynet_profiling=0):
    """
    :param tasks: list of TrainingTask instances. The first item takes on the role of the main task.
    :param stopping_criterion: "all": stop when all tasks signal stopping
                               "any" stop when "any" task signals stopping
                               integer n: stop when the n-th task signals stopping 
    :param dynet_profiling: if > 0, print computation graph
    """
    self.dynet_profiling = dynet_profiling
    self.stopping_criterion = stopping_criterion
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    for task in tasks[1:]:
      if not task.trainer is tasks[0].trainer:
        raise ValueError("Tasks must reference-share Trainer objects!")
    self.train = None
  def trigger_train_event(self, value):
    """
    Trigger set_train event, but only if that would lead to a change of the value
    of set_train.
    :param value: True or False
    """
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
  def __init__(self, yaml_context, tasks, stopping_criterion="all", dynet_profiling=0):
    super(JointMultiTrainingTask, self).__init__(tasks=tasks, 
                                                 stopping_criterion=stopping_criterion,
                                                 dynet_profiling=dynet_profiling)
    self.yaml_context = yaml_context
  def run_training(self, update_weights=True):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    self.trigger_train_event(update_weights)
    while True:
      task_losses = []
      for task, task_gen in task_generators.items():
        src, trg = next(task_gen)
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
      else:
        if self.tasks[self.stopping_criterion].should_stop_training(): break
  @property
  def corpus_parser(self):
    """
    Allow access to corpus_parser of main task
    """
    return self.tasks[0].corpus_parser
  @property
  def model(self):
    """
    Allow access to model of main task
    """
    return self.tasks[0].model
  

class SerialMultiTrainingTask(BaseMultiTrainingTask, Serializable):
  yaml_tag = u"!SerialMultiTrainingTask"
  """
  Multi-task training where training steps are performed one after another.
  The relative weight between tasks are explicitly specified explicitly, and for
  each step one task is drawn at random accordingly. 
  Compared to JointMultiTrainingTask, this class may save memory because models
  are only loaded individually. It also supports disabling training for some
  tasks by setting the task weight to 0.
  """
  def __init__(self, yaml_context, tasks, task_weights=None, stopping_criterion="all", dynet_profiling=0):
    super(SerialMultiTrainingTask, self).__init__(tasks=tasks, 
                                                 stopping_criterion=stopping_criterion,
                                                 dynet_profiling=dynet_profiling)
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks) 
    self.yaml_context = yaml_context
  def run_training(self, update_weights=True):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    self.trigger_train_event(update_weights)
    while True:
      cur_task = np.random.choice(self.tasks, p=self.task_weights)
      task_gen = task_generators[cur_task]
      src, trg = next(task_gen)
      task_loss = cur_task.training_step(src, trg)
      if update_weights:
        self.update_weights(task_loss, self.tasks[0].trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      if cur_task.checkpoint_needed():
        self.trigger_train_event(False)
        cur_task.checkpoint()
      self.trigger_train_event(update_weights)
      if self.stopping_criterion=="all":
        if all([task.should_stop_training() for task in self.tasks]): break
      elif self.stopping_criterion=="any":
        if any([task.should_stop_training() for task in self.tasks]): break
      else:
        if self.tasks[self.stopping_criterion].should_stop_training(): break
  @property
  def corpus_parser(self):
    """
    Allow access to corpus_parser of main task
    """
    return self.tasks[0].corpus_parser
  @property
  def model(self):
    """
    Allow access to model of main task
    """
    return self.tasks[0].model
  