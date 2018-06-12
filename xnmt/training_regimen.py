import argparse
from typing import Sequence
from collections import OrderedDict

from xnmt.settings import settings
import numpy as np
import dynet as dy

from xnmt.model_base import TrainableModel
from xnmt.loss_tracker import TrainLossTracker
from xnmt.loss_calculator import MLELoss
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, bare, Ref
from xnmt import training_task, optimizer, batcher

class TrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, save_fct, update_weights=True):
    """
    Run training steps in a loop until stopping criterion is reached.

    Args:
      save_fct: function to be invoked to save a model at dev checkpoints
      update_weights (bool): Whether parameters should be updated
    """
    raise NotImplementedError("")

  def backward(self, loss: dy.Expression, dynet_profiling: int) -> None:
    """
    Perform backward pass to accumulate gradients.

    Args:
      loss: Result of self.training_step(...)
      dynet_profiling: if > 0, print the computation graph
    """
    if dynet_profiling and dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()

  def update(self, trainer: optimizer.XnmtOptimizer) -> None:
    """
    Update DyNet weights using the given optimizer.

    Args:
      trainer: DyNet trainer
    """
    trainer.update()

class SimpleTrainingRegimen(training_task.SimpleTrainingTask, TrainingRegimen, Serializable):
  """
  Args:
    model (TrainableModel): the model
    src_file (str): the source training file
    trg_file (str): the target training file
    dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    dev_zero (bool): if True, add a checkpoint before training loop is entered (useful with pretrained models).
    batcher (Batcher): Type of batcher
    loss_calculator (LossCalculator): The method for calculating the loss.
    trainer (XnmtOptimizer): Trainer object, default is SGD with learning rate 0.1
    run_for_epochs (int):
    lr_decay (float):
    lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    patience (int): apply LR decay after dev scores haven't improved over this many checkpoints
    initial_patience (int): if given, allows adjusting patience for the first LR decay
    dev_tasks (List[EvalTask]): A list of tasks to use during the development stage.
    dev_combinator: A formula to combine together development scores into a single score to
                    choose whether to perform learning rate decay, etc.
                    e.g. 'x[0]-x[1]' would say that the first dev task score minus the
                    second dev task score is our measure of how good we're doing. If not
                    specified, only the score from the first dev task will be used.
    restart_trainer (bool): Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command (str): Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to 'true'.
    name (str): will be prepended to log outputs if given
    sample_train_sents (int):
    max_num_train_sents (int):
    max_src_len (int):
    max_trg_len (int):
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    commandline_args (Namespace):
  """
  yaml_tag = '!SimpleTrainingRegimen'

  @serializable_init
  def __init__(self, model=Ref("model"), src_file=None, trg_file=None, dev_every=0, dev_zero=False,
               batcher=bare(batcher.SrcBatcher, batch_size=32), loss_calculator=bare(MLELoss), trainer=None,
               run_for_epochs=None, lr_decay=1.0, lr_decay_times=3, patience=1, initial_patience=None, dev_tasks=None,
               dev_combinator=None, restart_trainer: bool = False, reload_command=None, name="{EXP}",
               sample_train_sents=None, max_num_train_sents=None, max_src_len=None, max_trg_len=None,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"),
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
                     dev_combinator=dev_combinator,
                     restart_trainer=restart_trainer,
                     reload_command=reload_command,
                     name=name,
                     sample_train_sents=sample_train_sents,
                     max_num_train_sents=max_num_train_sents,
                     max_src_len=max_src_len,
                     max_trg_len=max_trg_len)
    self.dev_zero = dev_zero
    self.trainer = trainer or optimizer.SimpleSGDTrainer(e0=0.1)
    self.dynet_profiling = getattr(commandline_args, "dynet_profiling", 0) if commandline_args else 0
    self.train_loss_tracker = TrainLossTracker(self)
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct, update_weights=True):
    """
    Main training loop (overwrites TrainingRegimen.run_training())
    """
    if self.run_for_epochs > 0:
      for src,trg in self.next_minibatch():
        if self.dev_zero:
          self.checkpoint_and_save(save_fct)
          self.dev_zero = False
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        with self.train_loss_tracker.time_tracker:
          self.model.set_train(True)
          loss_builder = self.training_step(src, trg)
          loss = loss_builder.compute()
          if update_weights:
            self.backward(loss, self.dynet_profiling)
            self.update(self.trainer)
        self.train_loss_tracker.report(trg, loss_builder.get_factored_loss_val(comb_method=self.loss_comb_method))
        if self.checkpoint_needed():
          self.checkpoint_and_save(save_fct)
        if self.should_stop_training(): break

  def checkpoint_and_save(self, save_fct):
    should_save = self.checkpoint()
    if should_save:
      save_fct()


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
    dev_zero (bool): if True, add a checkpoint before training loop is entered (useful with pretrained models).
    commandline_args (Namespace):
  """
  def __init__(self,
               tasks,
               trainer=None,
               dev_zero=False,
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    self.dynet_profiling = getattr(commandline_args, "dynet_profiling", 0) if commandline_args else 0
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    self.trainer = trainer or optimizer.SimpleSGDTrainer(e0=0.1)
    for task in tasks[1:]:
      if hasattr(task, "trainer") and task.trainer is not None:
        raise ValueError("Can instantiate only one trainer object. Possibly, multiple training regimens were created when training tasks should have been used.")
    self.train = None
    self.model_file = ParamManager.param_col.model_file
    for task in tasks:
      task.trainer = trainer
    self.dev_zero = dev_zero

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
  Multi-task training where gradients are accumulated and weight updates are thus performed jointly for each task.
  The relative weight between tasks can be configured by setting each tasks batch size accordingly.
  The stopping criterion of the first task is used (other tasks' stopping criteria are ignored).
  
  Args:
    tasks: training tasks
    trainer: the trainer is shared across tasks
    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained models).
    per_task_backward: if ``True``, call backward() for each task separately and renew computation graph between
                       tasks. Yields the same results, but ``True`` uses less memory while ``False`` may be
                       faster when using autobatching.
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    commandline_args:
  """
  yaml_tag = "!SameBatchMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks: Sequence[training_task.TrainingTask], trainer: optimizer.XnmtOptimizer = None,
               dev_zero: bool = False, per_task_backward: bool = True,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"),
               commandline_args: argparse.Namespace = Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, commandline_args=commandline_args)
    self.train_loss_trackers = {task : TrainLossTracker(task) for task in tasks}
    self.per_task_backward = per_task_backward
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct, update_weights=True):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    if self.tasks[0].run_for_epochs > 0:
      while True:
        task_losses = []
        task_src_trg = []
        for task, task_gen in task_generators.items():
          src, trg = next(task_gen)
          task_src_trg.append((task, src, trg))
        if self.dev_zero: # True only in first iteration
          self.checkpoint_and_save(save_fct)
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        task_trg_loss_stats = {}
        with self.train_loss_trackers[self.tasks[0]].time_tracker:
          self.trigger_train_event(True)
          for task, src, trg in task_src_trg:
            loss_builder = task.training_step(src, trg)
            task_trg_loss_stats[task] = (trg, loss_builder.get_factored_loss_val(comb_method=self.loss_comb_method))
            if self.per_task_backward:
              self.backward(loss_builder.compute(), self.dynet_profiling)
              dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
            else:
              task_losses.append(loss_builder.compute())
          if update_weights:
            if not self.per_task_backward:
              self.backward(sum(task_losses), self.dynet_profiling)
            self.update(self.trainer)
        for task, (trg, stats) in task_trg_loss_stats.items():
          self.train_loss_trackers[task].report(trg, stats)
        self.checkpoint_and_save(save_fct)
        if self.tasks[0].should_stop_training(): break

  def checkpoint_and_save(self, save_fct):
    for task_i, task in enumerate(self.tasks):
      if self.dev_zero or task.checkpoint_needed():
        should_save = task.checkpoint(control_learning_schedule=(task_i == 0))
        if should_save:
          save_fct()
    self.dev_zero = False


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
    dev_zero (bool): if True, add a checkpoint before training loop is entered (useful with pretrained models).
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    commandline_args (Namespace):
  """
  yaml_tag = "!AlternatingBatchMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks, task_weights=None, trainer=None, dev_zero=False,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"),
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, commandline_args=commandline_args)
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks)
    if len(self.task_weights) != len(self.tasks):
      raise ValueError(f"number of tasks must match number of task weights; "
                       f"found: {len(self.task_weights)} != {len(self.tasks)}")
    self.train_loss_trackers = {task: TrainLossTracker(task) for task in tasks}
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct, update_weights=True):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    dev_zero = {i:self.dev_zero for i in range(len(self.tasks))}
    if self.tasks[0].run_for_epochs > 0:
      while True:
        dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
        cur_task_i = np.random.choice(range(len(self.tasks)), p=self.task_weights)
        cur_task = self.tasks[cur_task_i]
        task_gen = task_generators[cur_task]
        src, trg = next(task_gen)
        if dev_zero[cur_task_i]: self.checkpoint_and_save(cur_task, cur_task_i, save_fct, dev_zero)
        cur_train_loss_tracker = self.train_loss_trackers[cur_task]
        with cur_train_loss_tracker.time_tracker:
          self.trigger_train_event(True)
          loss_builder = cur_task.training_step(src, trg)
          if update_weights:
            self.backward(loss=loss_builder.compute(), dynet_profiling=self.dynet_profiling)
            self.update(trainer=self.trainer)
        cur_train_loss_tracker.report(trg, loss_builder.get_factored_loss_val(comb_method=self.loss_comb_method))
        self.checkpoint_and_save(cur_task, cur_task_i, save_fct, dev_zero)
        if self.tasks[0].should_stop_training(): break

  def checkpoint_and_save(self, cur_task, cur_task_i, save_fct, dev_zero):
    if dev_zero[cur_task_i] or cur_task.checkpoint_needed():
      dev_zero[cur_task_i] = False
      should_save = cur_task.checkpoint(control_learning_schedule=(cur_task_i == 0))
      if should_save:
        save_fct()


class SerialMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  """
  Trains only first task until stopping criterion met, then the same for the
  second task, etc.

  Useful to realize a pretraining-finetuning strategy.

  Args:
    tasks (List[TrainingTask]): training tasks. The currently active task is treated as main task.
    trainer (XnmtOptimizer): the trainer is shared across tasks
    dev_zero (bool): if True, add a checkpoint before training loop is entered (useful with pretrained models).
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    commandline_args (Namespace):
  """

  yaml_tag = "!SerialMultiTaskTrainingRegimen"

  @serializable_init
  def __init__(self, tasks, trainer=None, dev_zero=False,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"),
               commandline_args=Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, commandline_args=commandline_args)
    self.train_loss_trackers = {task: TrainLossTracker(task) for task in tasks}
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct, update_weights=True):
    dev_zero = {i:self.dev_zero for i in range(len(self.tasks))}
    for cur_task_id in range(len(self.tasks)):
      self.train = None
      cur_task = self.tasks[cur_task_id]
      cur_train_loss_tracker = self.train_loss_trackers[cur_task]
      task_gen = cur_task.next_minibatch()
      if cur_task.run_for_epochs > 0:
        while True:
          dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
          src, trg = next(task_gen)
          if dev_zero[cur_task_id]: self.checkpoint_and_save(cur_task, cur_task_id, save_fct, dev_zero)
          with cur_train_loss_tracker.time_tracker:
            self.trigger_train_event(True)
            loss_builder = cur_task.training_step(src, trg)
            task_loss = loss_builder.compute()
            if update_weights:
              self.backward(task_loss, self.dynet_profiling)
              self.update(self.trainer)
          cur_train_loss_tracker.report(trg, loss_builder.get_factored_loss_val(comb_method=self.loss_comb_method))
          self.checkpoint_and_save(cur_task, cur_task_id, save_fct, dev_zero)
          if cur_task.should_stop_training(): break

  def checkpoint_and_save(self, cur_task, cur_task_id, save_fct, dev_zero):
    if dev_zero[cur_task_id] or cur_task.checkpoint_needed():
      dev_zero[cur_task_id] = False
      should_save = cur_task.checkpoint(control_learning_schedule=True)
      if should_save:
        save_fct()
