from collections import OrderedDict

from simple_settings import settings
import numpy as np
import dynet as dy

from xnmt.param_collection import ParamManager
from xnmt.serialize.serializable import Serializable, bare, Ref, Path
from xnmt.serialize.serializer import serializable_init
import xnmt.optimizer
from xnmt.training_task import SimpleTrainingTask

class TrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, save_fct, update_weights=True):
    """
    Runs training steps in a loop until stopping criterion is reached.

    Args:
      save_fct: function to be invoked to save a model at dev checkpoints
      update_weights (bool): Whether parameters should be updated
    """
    raise NotImplementedError("")
  def update_weights(self, loss, trainer, dynet_profiling):
    """
    Standardized way to perform backward pass and parameter updates.

    Args:
      loss: Result of self.training_step(...)
      trainer (XnmtOptimizer): DyNet trainer
      dynet_profiling (int): if > 0, print the computation graph
    """
    if dynet_profiling and dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()
    trainer.update()

class SimpleTrainingRegimen(SimpleTrainingTask, TrainingRegimen, Serializable):
  """
  Args:
    model (GeneratorModel): the model
    src_file (str): the source training file
    trg_file (str): the target training file
    dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    batcher (Batcher): Type of batcher
    loss_calculator (LossCalculator): The method for calculating the loss.
    trainer (XnmtOptimizer): Trainer object, default is SGD with learning rate 0.1
    run_for_epochs (int):
    lr_decay (float):
    lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    patience (int): apply LR decay after dev scores haven't improved over this many checkpoints
    initial_patience (int): if given, allows adjusting patience for the first LR decay
    dev_tasks (List[EvalTask]): A list of tasks to use during the development stage.
    restart_trainer (bool): Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command (str): Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to 'true'.
    name (str): will be prepended to log outputs if given
    sample_train_sents (int):
    max_num_train_sents (int):
    max_src_len (int):
    max_trg_len (int):
    commandline_args (Namespace):
  """
  yaml_tag = '!SimpleTrainingRegimen'

  @serializable_init
  def __init__(self, model=Ref("model"), src_file=None, trg_file=None,
               dev_every=0, batcher=bare(xnmt.batcher.SrcBatcher, batch_size=32),
               loss_calculator=None, trainer=None, run_for_epochs=None,
               lr_decay=1.0, lr_decay_times=3, patience=1, initial_patience=None,
               dev_tasks=None, restart_trainer: bool = False, reload_command=None,
               name="{EXP}", sample_train_sents=None, max_num_train_sents=None,
               max_src_len=None, max_trg_len=None,
               commandline_args=Ref("exp_global.commandline_args", default=None)):

    super().__init__(model=model,
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
                     name=name,
                     sample_train_sents=sample_train_sents,
                     max_num_train_sents=max_num_train_sents,
                     max_src_len=max_src_len,
                     max_trg_len=max_trg_len)
    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(e0=0.1)
    self.dynet_profiling = getattr(commandline_args, "dynet_profiling", 0) if commandline_args else 0

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

  Args:
    tasks (List[TrainingTask]): list of training tasks.
                The first item takes on the role of the main task, meaning it
                will control early stopping, learning rate schedule, and
                model checkpoints.
    trainer (XnmtOptimizer): Trainer object, default is SGD with learning rate 0.1
    commandline_args (Namespace):
  """
  def __init__(self,
               tasks,
               trainer=None,
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    self.dynet_profiling = getattr(commandline_args, "dynet_profiling", 0) if commandline_args else 0
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(e0=0.1)
    for task in tasks[1:]:
      if hasattr(task, "trainer") and task.trainer is not None:
        raise ValueError("Can instantiate only one trainer object. Possibly, multiple training regimens were created when training tasks should have been used.")
    self.train = None
    self.model_file = ParamManager.param_col.model_file
    for task in tasks:
      task.trainer = trainer

  def load_data(self):
    for task in self.tasks:
      task.load_data()

  def trigger_train_event(self, value):
    """
    Trigger set_train event, but only if that would lead to a change of the value
    of set_train.
    Args:
      value: True or False
    """
    if self.train is None:
      self.train = value
      self.tasks[0].model.set_train(value) # tasks[0] is arbitrary; will invoke on_set_train() for all models
    else:
      if value!=self.train:
        self.train = value
        self.tasks[0].model.set_train(value)

class SameBatchMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  """
  Multi-task training where gradients are accumulated and weight updates
  are thus performed jointly for each task. The relative weight between
  tasks can be configured by setting each tasks batch size accordingly.
  The stopping criterion of the first task is used (other tasks' stopping criteria are ignored).
  
  Args:
    tasks (List[TrainingTask]): training tasks
    trainer (XnmtOptimizer): the trainer is shared across tasks
    commandline_args (Namespace):
  """
  yaml_tag = "!SameBatchMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks, trainer=None,
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, commandline_args=commandline_args)
  def run_training(self, save_fct, update_weights=True):
    self.load_data()
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
  """
  Multi-task training where training steps are performed one after another.
  The relative weight between tasks are explicitly specified explicitly, and for
  each step one task is drawn at random accordingly.
  Compared to JointMultiTaskTrainingRegimen, this class may save memory because models
  are only loaded individually. It also supports disabling training for some
  tasks by setting the task weight to 0.
  The stopping criterion of the first task is used (other tasks' stopping criteria are ignored).

  Args:
    tasks (List[TrainingTask]): training tasks
    trainer (XnmtOptimizer): the trainer is shared across tasks
    commandline_args (Namespace):
  """
  yaml_tag = "!AlternatingBatchMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks, task_weights=None, trainer=None,
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, commandline_args=commandline_args)
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks)
  def run_training(self, save_fct, update_weights=True):
    self.load_data()
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

  Args:
    tasks (List[TrainingTask]): training tasks. The currently active task is treated as main task.
    trainer (XnmtOptimizer): the trainer is shared across tasks
    commandline_args (Namespace):
  """

  yaml_tag = "!SerialMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks, trainer=None, commandline_args=Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, commandline_args=commandline_args)
  def run_training(self, save_fct, update_weights=True):
    self.load_data()
    for cur_task_id in range(len(self.tasks)):
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
