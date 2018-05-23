from subprocess import Popen
from asteval import Interpreter
import random
import numpy as np
from typing import Optional

from xnmt import batcher, events, model_base, input_reader, logger, loss, loss_tracker, loss_calculator, param_collection
from xnmt.persistence import serializable_init, Serializable, bare

class TrainingTask(object):
  """
  Base class for a training task. Training tasks can perform training steps
  and keep track of the training state, but may not implement the actual training
  loop.
  """
  def __init__(self, model):
    self.model = model

  def load_data(self):
    """
    Used to load data.
    """
    raise NotImplementedError("")
  def should_stop_training(self):
    """
    Returns:
      True iff training is finished, i.e. training_step(...) should not be called again
    """
    raise NotImplementedError("")

  def training_step(self, src, trg):
    """
    Performs forward pass corresponding to a single training step.
    Training logic like switching epochs, reshuffling batches, etc. must be
    handled as well.

    Args:
      src: src minibatch
      trg: trg minibatch
    Returns:
      Loss
    """
    raise NotImplementedError("")

  def checkpoint_needed(self):
    raise NotImplementedError()

  def checkpoint(self, control_learning_schedule=False, out_ext=".dev_hyp", ref_ext=".dev_ref",
                 encoding='utf-8'):
    """
    Performs a dev checkpoint

    Args:
      control_learning_schedule: If False, only evaluate dev data.
                                      If True, also perform model saving, LR decay etc. if needed.
      out_ext:
      ref_ext:
      encoding:
    Returns:
      True if the model needs saving, False otherwise
    """
    raise NotImplementedError()


class SimpleTrainingTask(TrainingTask, Serializable):
  """
  Args:
    model (model_base.TrainableModel): a trainable model
    src_file: The file for the source data.
    trg_file: The file for the target data.
    dev_every (int): dev checkpoints every n sentences (0 for only after epoch)
    batcher: Type of batcher
    loss_calculator:
    run_for_epochs (int): number of epochs (None for unlimited epochs)
    lr_decay (float):
    lr_decay_times (int):  Early stopping after decaying learning rate a certain number of times
    patience (int): apply LR decay after dev scores haven't improved over this many checkpoints
    initial_patience (int): if given, allows adjusting patience for the first LR decay
    dev_tasks: A list of tasks to run on the development set
    dev_combinator: A formula to combine together development scores into a single score to
                    choose whether to perform learning rate decay, etc.
                    e.g. 'x[0]-x[1]' would say that the first dev task score minus the
                    second dev task score is our measure of how good we're doing. If not
                    specified, only the score from the first dev task will be used.
    restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command: Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to 'true'.
    sample_train_sents: If given, load a random subset of training sentences before each epoch. Useful when training data does not fit in memory.
    max_num_train_sents:
    max_src_len:
    max_trg_len:
    name: will be prepended to log outputs if given
  """
  yaml_tag = '!SimpleTrainingTask'

  @serializable_init
  def __init__(self, model, src_file=None, trg_file=None, dev_every=0,
               batcher=bare(batcher.SrcBatcher, batch_size=32), loss_calculator=bare(loss_calculator.MLELoss),
               run_for_epochs=None, lr_decay=1.0, lr_decay_times=3, patience=1,
               initial_patience=None, dev_tasks=None, dev_combinator=None, restart_trainer=False,
               reload_command=None, name=None, sample_train_sents: Optional[int] = None,
               max_num_train_sents=None, max_src_len=None, max_trg_len=None):
    self.src_file = src_file
    self.trg_file = trg_file
    self.dev_tasks = dev_tasks
    self.dev_combinator = dev_combinator

    if lr_decay > 1.0 or lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.lr_decay = lr_decay
    self.patience = patience
    self.initial_patience = initial_patience
    self.lr_decay_times = lr_decay_times
    self.restart_trainer = restart_trainer
    self.run_for_epochs = run_for_epochs

    self.early_stopping_reached = False
    # training state
    self.training_state = TrainingState()

    self.reload_command = reload_command

    self.model = model
    self.loss_calculator = loss_calculator

    self.sample_train_sents = sample_train_sents
    self.max_num_train_sents = max_num_train_sents
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len

    self.batcher = batcher
    self.dev_loss_tracker = loss_tracker.DevLossTracker(self, dev_every, name)
    self.name = name

  def _augment_data_initial(self):
    """
    Called before loading corpus for the first time, if reload_command is given
    """
    augment_command = self.reload_command
    logger.debug('initial augmentation')
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch 0", shell=True)
      self._augmentation_handle.wait()

  def _augment_data_next_epoch(self):
    """
    This is run in the background if reload_command is given to prepare data for the next epoch
    """
    augment_command = self.reload_command
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
      self._augmentation_handle.wait()

    self._augmentation_handle.poll()
    retcode = self._augmentation_handle.returncode
    if retcode is not None:
      if self.training_state.epoch_num > 0:
        logger.info('using reloaded data')
      # reload the data   
      self.src_data, self.trg_data, self.src_batches, self.trg_batches = \
          input_reader.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                          self.src_file, self.trg_file,
                                          batcher=self.batcher, sample_sents=self.sample_train_sents,
                                          max_num_sents=self.max_num_train_sents,
                                          max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
      # restart data generation
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
    else:
      logger.info('new data set is not ready yet, using data from last epoch.')

  @events.register_xnmt_event
  def new_epoch(self, training_task, num_sents):
    """
    New epoch event.

    Args:
      training_task: Indicates which training task is advancing to the next epoch.
      num_sents: Number of sentences in the upcoming epoch (may change between epochs)
    """
    pass

  def should_stop_training(self):
    """
    Signal stopping if self.early_stopping_reached is marked or we exhausted the number of requested epochs.
    """
    return self.early_stopping_reached \
      or self.run_for_epochs is not None and (self.training_state.epoch_num > self.run_for_epochs
                                              or (self.training_state.epoch_num == self.run_for_epochs and
                                                  self.training_state.steps_into_epoch >= self.cur_num_minibatches()))

  def cur_num_minibatches(self):
    """
    Current number of minibatches (may change between epochs, e.g. for randomizing batchers or if reload_command is given)
    """
    return len(self.src_batches)

  def cur_num_sentences(self):
    """
    Current number of parallel sentences (may change between epochs, e.g. if reload_command is given)
    """
    return len(self.src_data)

  def advance_epoch(self):
    """
    Shifts internal state to the next epoch, including data (re-)loading, batch re-packing and shuffling.
    """
    if self.reload_command is not None:
      if self.training_state.epoch_num==0:
        self._augmentation_handle = None
        self._augment_data_initial()
      else:
        self._augment_data_next_epoch()
    if self.training_state.epoch_num==0 or self.sample_train_sents:
      self.src_data, self.trg_data, self.src_batches, self.trg_batches = \
        input_reader.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                               self.src_file, self.trg_file,
                                               batcher=self.batcher, sample_sents=self.sample_train_sents,
                                               max_num_sents=self.max_num_train_sents,
                                               max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
    self.training_state.epoch_seed = random.randint(1,2147483647)
    random.seed(self.training_state.epoch_seed)
    np.random.seed(self.training_state.epoch_seed)
    self.src_batches, self.trg_batches = \
      self.batcher.pack(self.src_data, self.trg_data)
    self.training_state.epoch_num += 1
    self.training_state.steps_into_epoch = 0
    self.training_state.sents_into_epoch = 0
    self.minibatch_order = list(range(0, self.cur_num_minibatches()))
    np.random.shuffle(self.minibatch_order)
    self.new_epoch(training_task=self, num_sents=self.cur_num_sentences())

  def next_minibatch(self):
    """
    Infinitely loops over training minibatches and calls advance_epoch() after every complete sweep over the corpus.

    Returns:
      Generator yielding (src_batch,trg_batch) tuples
    """
    while True:
      self.advance_epoch()
      for batch_num in self.minibatch_order:
        src = self.src_batches[batch_num]
        trg = self.trg_batches[batch_num]
        self.training_state.steps_into_epoch += 1
        self.training_state.sents_into_epoch += len(src)
        self.training_state.sents_since_start += len(src)
        yield src, trg

  def training_step(self, src, trg):
    """
    Performs forward pass, backward pass, parameter update for the given minibatch
    """
    loss_builder = loss.LossBuilder()
    standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
    additional_loss = self.model.calc_additional_loss(standard_loss)
    loss_builder.add_loss("standard_loss", standard_loss)
    loss_builder.add_loss("additional_loss", additional_loss)
    return loss_builder

  def checkpoint_needed(self):
    return self.dev_loss_tracker.should_report_dev()

  def checkpoint(self, control_learning_schedule=True):
    """
    Performs a dev checkpoint

    Args:
      control_learning_schedule: If False, only evaluate dev data.
                                      If True, also perform model saving, LR decay etc. if needed.
    Returns:
      True if the model needs saving, False otherwise
    """
    # Perform evaluation
    if self.dev_tasks and len(self.dev_tasks) > 0:
      dev_scores = []
      with self.dev_loss_tracker.time_tracker:
        logger.info("> Checkpoint")
        for dev_task in self.dev_tasks:
          dev_score, dev_word_cnt = dev_task.eval()
          if type(dev_score) == list:
            dev_scores.extend(dev_score)
          else:
            dev_scores.append(dev_score)
        self.dev_loss_tracker.set_dev_score(dev_word_cnt, dev_scores[0])
        for dev_score in dev_scores[1:]:
          self.dev_loss_tracker.add_aux_score(dev_score)
      self.dev_loss_tracker.report()

      # Control the learning schedule
      if control_learning_schedule:
        # Check if this is the best
        is_best = False
        if self.dev_combinator is not None:
          x = [y.value() for y in dev_scores]
          aevala = Interpreter()
          my_score = aevala(self.dev_combinator)
          logger.info('  combined dev scores according to {}: {}'.format(self.dev_combinator, my_score))
          if self.training_state.best_dev_score is None or my_score > self.training_state.best_dev_score:
            self.training_state.best_dev_score = my_score
            is_best = True
        elif dev_scores[0].better_than(self.training_state.best_dev_score):
          self.training_state.best_dev_score = dev_scores[0]
          is_best = True
        # If this is the best, write the model out
        if is_best:
          self.training_state.cur_attempt = 0
          needs_saving = True
          logger.info(f"  best dev score, writing out model")
        else:
          needs_saving = False
          # otherwise: learning rate decay / early stopping
          self.training_state.cur_attempt += 1
          if self.lr_decay < 1.0:
            should_decay = False
            if (self.initial_patience is None or self.training_state.num_times_lr_decayed>0) \
                    and self.training_state.cur_attempt >= self.patience:
              should_decay = True
            if self.initial_patience is not None and self.training_state.num_times_lr_decayed==0 \
                    and self.training_state.cur_attempt >= self.initial_patience:
              should_decay = True
            if should_decay:
              self.training_state.num_times_lr_decayed += 1
              if self.training_state.num_times_lr_decayed > self.lr_decay_times:
                logger.info('  Early stopping')
                self.early_stopping_reached = True
              else:
                self.training_state.cur_attempt = 0
                self.trainer.learning_rate *= self.lr_decay
                logger.info('  new learning rate: %s' % self.trainer.learning_rate)
                if self.restart_trainer:
                  logger.info('  restarting trainer and reverting learned weights to best checkpoint..')
                  self.trainer.restart()
                  param_collection.ParamManager.param_col.revert_to_best_model()
      else: # case of not controling learning schedule
        needs_saving = False
    else: # case of no dev tasks
      needs_saving = True

    return needs_saving

class TrainingState(object):
  """
  This holds the state of the training loop.
  """
  def __init__(self):
    self.num_times_lr_decayed = 0
    self.cur_attempt = 0
    self.epoch_num = 0
    self.steps_into_epoch = 0
    self.sents_since_start = 0
    self.sents_into_epoch = 0
    self.best_dev_score = None
    # used to pack and shuffle minibatches (keeping track might help resuming crashed trainings in the future)
    self.epoch_seed = random.randint(1,2147483647)
