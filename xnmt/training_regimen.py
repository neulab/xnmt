# coding: utf-8
from __future__ import division, print_function

import argparse
import sys
import six
from six.moves import range
from collections import OrderedDict

import dynet as dy

# all Serializable objects must be imported here, otherwise we get in trouble with the
# YAML parser
import xnmt.batcher
from xnmt.embedder import *
from xnmt.attender import *
from xnmt.input import *
import xnmt.lstm
import xnmt.pyramidal
import xnmt.conv
import xnmt.ff
import xnmt.segment_transducer
import xnmt.residual
import xnmt.training_task
from xnmt.specialized_encoders import *
from xnmt.decoder import *
from xnmt.translator import *
from xnmt.retriever import *
from xnmt.corpus import *
from xnmt.segmenting_encoder import *
from xnmt.optimizer import SimpleSGDTrainer
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.serializer import YamlSerializer, Serializable
from xnmt.inference import SimpleInference
import xnmt.xnmt_evaluate
import xnmt.optimizer
from xnmt.training_task import SimpleTrainingTask

class TrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, update_weights=True):
    """
    Runs training steps in a loop until stopping criterion is reached.
    
    :param update_weights: Whether parameters should be updated
    """
    raise NotImplementedError("")

class SimpleTrainingRegimen(SimpleTrainingTask, TrainingRegimen, Serializable):
  yaml_tag = u'!SimpleTrainingRegimen'
  def __init__(self, yaml_context, corpus_parser, model, glob={},
               dev_every=0, batcher=None, loss_calculator=None, 
               pretrained_model_file="", src_format="text", trainer=None, 
               run_for_epochs=None, lr_decay=1.0, lr_decay_times=3, patience=1,
               initial_patience=None, dev_metrics="", schedule_metric="loss",
               restart_trainer=False, reload_command=None, dynet_profiling=0,
               name=None, inference=None):
    """
    :param yaml_context:
    :param corpus_parser: an input.InputReader object
    :param model: a generator.GeneratorModel object
    :param dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    :param batcher: Type of batcher. Defaults to SrcBatcher of batch size 32.
    :param loss_calculator:
    :param pretrained_model_file: Path of pre-trained model file
    :param src_format: Format of input data: text/contvec
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    :param lr_decay (float):
    :param lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    :param patience (int): apply LR decay after dev scores haven't improved over this many checkpoints
    :param initial_patience (int): if given, allows adjusting patience for the first LR decay
    :param dev_metrics: Comma-separated list of evaluation metrics (bleu/wer/cer)
    :param schedule_metric: determine learning schedule based on this dev_metric (loss/bleu/wer/cer)
    :param restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    :param reload_command: Command to change the input data after each epoch.
                           --epoch EPOCH_NUM will be appended to the command.
                           To just reload the data after each epoch set the command to 'true'.
    :param dynet_profiling:
    :param name: will be prepended to log outputs if given
    :param inference: used for inference during dev checkpoints if dev_metrics are specified
    """
    super(SimpleTrainingRegimen, self).__init__(yaml_context=yaml_context,
                                                corpus_parser=corpus_parser,
                                                model=model,
                                                glob=glob,
                                                dev_every=dev_every,
                                                batcher=batcher,
                                                loss_calculator=loss_calculator, 
                                                pretrained_model_file=pretrained_model_file,
                                                src_format=src_format,
                                                run_for_epochs=run_for_epochs,
                                                lr_decay=lr_decay,
                                                lr_decay_times=lr_decay_times,
                                                patience=patience,
                                                initial_patience=initial_patience,
                                                dev_metrics=dev_metrics,
                                                schedule_metric=schedule_metric,
                                                restart_trainer=restart_trainer,
                                                reload_command=reload_command,
                                                name=name,
                                                inference=inference)
    self.trainer = trainer or xnmt.optimizer.SimpleSGDTrainer(self.yaml_context, 0.1)
    self.dynet_profiling = dynet_profiling

  def run_training(self, update_weights=True):
    """
    Main training loop (overwrites TrainingRegimen.run_training())
    """
    self.model.set_train(update_weights)
    for src,trg in self.next_minibatch():
      dy.renew_cg()
      loss = self.training_step(src, trg)
      if update_weights: self.update_weights(loss, self.trainer, self.dynet_profiling)
      if self.checkpoint_needed():
        if update_weights: self.model.set_train(False)
        should_save = self.checkpoint()
        if should_save:
          self.yaml_serializer.save_to_file(self.model_file, self,
                                        self.yaml_context.dynet_param_collection)
        if update_weights: self.model.set_train(True)
      if self.should_stop_training(): break

class MultiTaskTrainingRegimen(TrainingRegimen):
  """
  Base class for multi-task training classes.
  Mainly initializes tasks, performs sanity-checks, and manages set_train events.
  """
  def __init__(self, yaml_context, tasks, trainer=None, dynet_profiling=0):
    """
    :param tasks: list of TrainingTask instances.
                  The first item takes on the role of the main task, meaning it
                  will control early stopping, learning rate schedule, and
                  model checkpoints.
    :param trainer: Trainer object, default is SGD with learning rate 0.1
    :param dynet_profiling: if > 0, print computation graph
    """
    self.dynet_profiling = dynet_profiling
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    self.trainer = trainer or SimpleSGDTrainer(self.yaml_context, 0.1)
    for task in tasks[1:]:
      if hasattr(task, "trainer") and task.trainer is not None:
        raise ValueError("Can instantiate only one trainer object. Possibly, multiple training regimens were created when training tasks should have been used.")
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

class JointMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  yaml_tag = u"!JointMultiTaskTrainingRegimen"
  """
  Multi-task training where gradients are accumulated and weight updates
  are thus performed jointly for each task. The relative weight between
  tasks can be configured by setting each tasks batch size accordingly.
  """
  def __init__(self, yaml_context, tasks, trainer=None, dynet_profiling=0):
    super(JointMultiTaskTrainingRegimen, self).__init__(yaml_context,
                                                 tasks=tasks, trainer=trainer,
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
        self.update_weights(sum(task_losses), self.trainer, self.dynet_profiling)
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
  

class SerialMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, Serializable):
  yaml_tag = u"!SerialMultiTaskTrainingRegimen"
  """
  Multi-task training where training steps are performed one after another.
  The relative weight between tasks are explicitly specified explicitly, and for
  each step one task is drawn at random accordingly. 
  Compared to JointMultiTaskTrainingRegimen, this class may save memory because models
  are only loaded individually. It also supports disabling training for some
  tasks by setting the task weight to 0.
  """
  def __init__(self, yaml_context, tasks, task_weights=None, trainer=None, dynet_profiling=0):
    super(SerialMultiTaskTrainingRegimen, self).__init__(yaml_context,
                                                  tasks=tasks, trainer=trainer,
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
        self.update_weights(task_loss, self.trainer, self.dynet_profiling)
      if update_weights: self.tasks[0].model.set_train(False)
      if cur_task.checkpoint_needed():
        self.trigger_train_event(False)
        should_save = cur_task.checkpoint(control_learning_schedule = (cur_task_i==0))
        if should_save:
          self.yaml_serializer.save_to_file(self.model_file, self,
                                            self.yaml_context.dynet_param_collection)
      self.trigger_train_event(update_weights)
      if self.tasks[0].should_stop_training(): break

