from collections import OrderedDict

from simple_settings import settings
import numpy as np
import dynet as dy

from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
import xnmt.optimizer
from xnmt.training_task import SimpleTrainingTask

class TrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, save_fct, update_weights=True):
    """
    Runs training steps in a loop until stopping criterion is reached.
    
    :param save_fct: function to be invoked to save a model at dev checkpoints
    :param update_weights: Whether parameters should be updated
    """
    raise NotImplementedError("")
  def update_weights(self, loss, trainer, dynet_profiling):
    """
    Standardized way to perform backward pass and parameter updates.
    
    :param loss: Result of self.training_step(...)
    :param trainer: DyNet trainer / xnmt.optimizer object
    :param dynet_profiling: if > 0, print the computation graph 
    """
    if dynet_profiling and dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()
    trainer.update()

class SimpleTrainingRegimen(SimpleTrainingTask, TrainingRegimen, Serializable):
  yaml_tag = '!SimpleTrainingRegimen'
  def __init__(self, xnmt_global=Ref(Path("xnmt_global")), model=Ref(path=Path("model")),
               src_file=None, trg_file=None,
               dev_every=0, batcher=xnmt.batcher.SrcBatcher(32), loss_calculator=None, 
               trainer=None, run_for_epochs=None, lr_decay=1.0, lr_decay_times=3,
               patience=1, initial_patience=None, dev_tasks=None,
               restart_trainer=False, reload_command=None, name=None):
    """
    :param xnmt_global:
    :param model: a generator.GeneratorModel object
    :param src_file: the source training file
    :param trg_file: the target training file
    :param dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    :param batcher: Type of batcher
    :param loss_calculator: The method for calculating the loss.
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    :param lr_decay (float):
    :param lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    :param patience (int): apply LR decay after dev scores haven't improved over this many checkpoints
    :param initial_patience (int): if given, allows adjusting patience for the first LR decay
    :param dev_tasks: A list of tasks to use during the development stage.
    :param restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    :param reload_command: Command to change the input data after each epoch.
                           --epoch EPOCH_NUM will be appended to the command.
                           To just reload the data after each epoch set the command to 'true'.
    :param name: will be prepended to log outputs if given
    """
    super().__init__(xnmt_global=xnmt_global,
                     model=model,
                     src_file=src_file,
                     trg_file=trg_file,
                     dev_every=dev_every,
                     batcher=batcher,
                     loss_calculator=loss_calculator, 
                     run_for_epochs=run_for_epochs,
                     lr_decay=lr_decay,
                     lr_decay_times=lr_decay_times,
                     patience=patience,
                     initial_patience=initial_patience,
                     dev_tasks=dev_tasks,
                     restart_trainer=restart_trainer,
                     reload_command=reload_command,
                     name=name)
    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(xnmt_global=self.xnmt_global, e0=0.1)
    self.dynet_profiling = getattr(xnmt_global.commandline_args, "dynet_profiling", 0)

  def run_training(self, save_fct, update_weights=True):
    """
    Main training loop (overwrites TrainingRegimen.run_training())
    """
    self.load_data()
    self.model.set_train(update_weights)
    for src,trg in self.next_minibatch():
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      loss = self.training_step(src, trg)
      if update_weights: self.update_weights(loss, self.trainer, self.dynet_profiling)
      if self.checkpoint_needed():
        if update_weights: self.model.set_train(False)
        should_save = self.checkpoint()
        if should_save:
          save_fct()
        if update_weights: self.model.set_train(True)
      if self.should_stop_training(): break

class MultiTaskTrainingRegimen(TrainingRegimen):
  """
  Base class for multi-task training classes.
  Mainly initializes tasks, performs sanity-checks, and manages set_train events.
  """
  def __init__(self, tasks, trainer=None, xnmt_global=Ref(Path("xnmt_global"))):
    """
    :param tasks: list of TrainingTask instances.
                  The first item takes on the role of the main task, meaning it
                  will control early stopping, learning rate schedule, and
                  model checkpoints.
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    """
    self.dynet_profiling = xnmt_global.commandline_args.dynet_profiling
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(xnmt_global=self.xnmt_global, e0=0.1)
    for task in tasks[1:]:
      if hasattr(task, "trainer") and task.trainer is not None:
        raise ValueError("Can instantiate only one trainer object. Possibly, multiple training regimens were created when training tasks should have been used.")
    self.train = None
    self.model_file = xnmt_global.dynet_param_collection.model_file
    self.main_task = 0
    for task in tasks: task.trainer = trainer
  def init_data_vocabs(self):
    for task in self.tasks:
      task.load_data()
    for task in self.tasks:
      task.fix_vocabs()
    
  def trigger_train_event(self, value):
    """
    Trigger set_train event, but only if that would lead to a change of the value
    of set_train.
    :param value: True or False
    """
    if self.train is None:
      self.train = value
      self.tasks[0].model.set_train(value) # tasks[0] is arbitrary; will invoke on_set_train() for all models
    else:
      if value!=self.train:
        self.train = value
        self.tasks[0].model.set_train(value)
  @property
  def model(self):
    """
    Allow access to model of main task
    """
    return self.tasks[self.main_task].model
  @property
  def batcher(self):
    """
    Allow access to batcher of main task
    """
    return self.tasks[self.main_task].batcher

class SameBatchMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  yaml_tag = "!SameBatchMultiTaskTrainingRegimen"
  """
  Multi-task training where gradients are accumulated and weight updates
  are thus performed jointly for each task. The relative weight between
  tasks can be configured by setting each tasks batch size accordingly.
  """
  def __init__(self, tasks, trainer=None, xnmt_global=Ref(Path("xnmt_global"))):
    super().__init__(xnmt_global=xnmt_global, tasks=tasks, trainer=trainer)
    self.xnmt_global = xnmt_global
  def run_training(self, save_fct, update_weights=True):
    self.init_data_vocabs()
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    self.trigger_train_event(update_weights)
    while True:
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      task_losses = []
      for task, task_gen in task_generators.items():
        src, trg = next(task_gen)
        task_losses.append(task.training_step(src, trg))
      if update_weights:
        self.update_weights(sum(task_losses), self.trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      for task_i, task in enumerate(self.tasks):
        if task.checkpoint_needed():
          self.trigger_train_event(False)
          should_save = task.checkpoint(control_learning_schedule = (task_i==0) )
          if should_save:
            save_fct()
      self.trigger_train_event(update_weights)
      if self.tasks[0].should_stop_training(): break
  

class AlternatingBatchMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  yaml_tag = "!AlternatingBatchMultiTaskTrainingRegimen"
  """
  Multi-task training where training steps are performed one after another.
  The relative weight between tasks are explicitly specified explicitly, and for
  each step one task is drawn at random accordingly. 
  Compared to JointMultiTaskTrainingRegimen, this class may save memory because models
  are only loaded individually. It also supports disabling training for some
  tasks by setting the task weight to 0.
  """
  def __init__(self, tasks, task_weights=None, trainer=None, xnmt_global=Ref(Path("xnmt_global"))):
    super().__init__(xnmt_global=xnmt_global, tasks=tasks, trainer=trainer)
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks) 
    self.xnmt_global = xnmt_global
  def run_training(self, save_fct, update_weights=True):
    self.init_data_vocabs()
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    self.trigger_train_event(update_weights)
    while True:
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      cur_task_i = np.random.choice(range(len(self.tasks)), p=self.task_weights)
      cur_task = self.tasks[cur_task_i]
      task_gen = task_generators[cur_task]
      src, trg = next(task_gen)
      task_loss = cur_task.training_step(src, trg)
      if update_weights:
        self.update_weights(task_loss, self.trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      if cur_task.checkpoint_needed():
        self.trigger_train_event(False)
        should_save = cur_task.checkpoint(control_learning_schedule = (cur_task_i==0))
        if should_save:
          save_fct()
      self.trigger_train_event(update_weights)
      if self.tasks[0].should_stop_training(): break

class SerialMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  """
  Trains only first task until stopping criterion met, then the same for the
  second task, etc.
  
  Useful to realize a pretraining-finetuning strategy.
  """

  yaml_tag = "!SerialMultiTaskTrainingRegimen"
  
  def __init__(self, xnmt_global, tasks, trainer=None):
    """
    :param tasks: list of TrainingTask instances. The currently active task is treated as main task.
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    """
    super().__init__(xnmt_global=xnmt_global, tasks=tasks, trainer=trainer)
    self.xnmt_global = xnmt_global
  def run_training(self, save_fct, update_weights=True):
    self.init_data_vocabs()
    for cur_task_id in range(len(self.tasks)):
      self.main_task = cur_task_id
      self.train = None
      cur_task = self.tasks[cur_task_id]
      task_gen = cur_task.next_minibatch()
      self.trigger_train_event(update_weights)
      while True:
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        src, trg = next(task_gen)
        task_loss = cur_task.training_step(src, trg)
        if update_weights:
          self.update_weights(task_loss, self.trainer, self.dynet_profiling)
        if cur_task.checkpoint_needed():
          self.trigger_train_event(False)
          should_save = cur_task.checkpoint(control_learning_schedule = True)
          if should_save:
            save_fct()
        self.trigger_train_event(update_weights)
        if cur_task.should_stop_training(): break
