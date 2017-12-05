from __future__ import division, generators

from collections import OrderedDict
import numpy as np
import dynet as dy
from xnmt.serializer import Serializable, YamlSerializer


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

  def checkpoint(self, control_learning_schedule=False, out_ext=".dev_hyp", ref_ext=".dev_ref", 
                 encoding='utf-8'):
    """
    Performs a dev checkpoint
    :param control_learning_schedule: If False, only evaluate dev data.
                                      If True, also perform model saving, LR decay etc. if needed.
    :param out_ext:
    :param ref_ext:
    :param encoding:
    :returns: True if the model needs saving, False otherwise
    """
    raise NotImplementedError()

class BaseMultiTrainingTask(BaseTrainingRegimen, TrainingTask):
  """
  Base class for multi-task training classes.
  Mainly initializes tasks, performs sanity-checks, and manages set_train events.
  """
  def __init__(self, yaml_context, tasks, dynet_profiling=0):
    """
    :param tasks: list of TrainingTask instances.
                  The first item takes on the role of the main task, meaning it
                  will control early stopping, learning rate schedule, and
                  model checkpoints.
    :param dynet_profiling: if > 0, print computation graph
    """
    self.dynet_profiling = dynet_profiling
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    for task in tasks[1:]:
      if not task.trainer is tasks[0].trainer:
        raise ValueError("Tasks must reference-share Trainer objects!")
    self.train = None
    self.yaml_serializer = YamlSerializer()
    self.model_file = yaml_context.dynet_param_collection.model_file
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
  @property
  def xnmt_decoder(self):
    return self._xnmt_decoder
  @xnmt_decoder.setter
  def xnmt_decoder(self, value):
    self._xnmt_decoder = value
    for task in self.tasks:
      task.xnmt_decoder = value

class JointMultiTrainingTask(BaseMultiTrainingTask, Serializable):
  yaml_tag = u"!JointMultiTrainingTask"
  """
  Multi-task training where gradients are accumulated and weight updates
  are thus performed jointly for each task. The relative weight between
  tasks can be configured by setting each tasks batch size accordingly.
  """
  def __init__(self, yaml_context, tasks, dynet_profiling=0):
    super(JointMultiTrainingTask, self).__init__(yaml_context,
                                                 tasks=tasks, 
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
      for task_i, task in enumerate(self.tasks):
        if task.checkpoint_needed():
          self.trigger_train_event(False)
          should_save = task.checkpoint(control_learning_schedule = (task_i==0) )
          if should_save:
            self.yaml_serializer.save_to_file(self.model_file, self,
                                          self.yaml_context.dynet_param_collection)
      self.trigger_train_event(update_weights)
      if self.tasks[0].should_stop_training(): break
  

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
  def __init__(self, yaml_context, tasks, task_weights=None, dynet_profiling=0):
    super(SerialMultiTrainingTask, self).__init__(yaml_context,
                                                  tasks=tasks, 
                                                  dynet_profiling=dynet_profiling)
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks) 
    self.yaml_context = yaml_context
  def run_training(self, update_weights=True):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    self.trigger_train_event(update_weights)
    while True:
      cur_task_i = np.random.choice(range(len(self.tasks)), p=self.task_weights)
      cur_task = self.tasks[cur_task_i]
      task_gen = task_generators[cur_task]
      src, trg = next(task_gen)
      task_loss = cur_task.training_step(src, trg)
      if update_weights:
        self.update_weights(task_loss, self.tasks[0].trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      if cur_task.checkpoint_needed():
        self.trigger_train_event(False)
        should_save = cur_task.checkpoint(control_learning_schedule = (cur_task_i==0))
        if should_save:
          self.yaml_serializer.save_to_file(self.model_file, self,
                                            self.yaml_context.dynet_param_collection)
      self.trigger_train_event(update_weights)
      if self.tasks[0].should_stop_training(): break
