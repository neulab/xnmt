from subprocess import Popen
import random
import numpy as np

from xnmt import logger
from xnmt.batcher import SrcBatcher
from xnmt.events import register_xnmt_event
import xnmt.input_reader
from xnmt.loss import LossBuilder
from xnmt.loss_calculator import LossCalculator, MLELoss
from xnmt.loss_tracker import BatchLossTracker
from xnmt.param_collection import ParamManager
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
    model: a generator.GeneratorModel object
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
    restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command: Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to 'true'.
    sample_train_sents:
    max_num_train_sents:
    max_src_len:
    max_trg_len:
    name: will be prepended to log outputs if given
  """
  yaml_tag = '!SimpleTrainingTask'

  @serializable_init
  def __init__(self, model, src_file=None, trg_file=None, dev_every=0,
               batcher=bare(SrcBatcher, batch_size=32), loss_calculator=None,
               run_for_epochs=None, lr_decay=1.0, lr_decay_times=3, patience=1,
               initial_patience=None, dev_tasks=None, restart_trainer=False,
               reload_command=None, name=None, sample_train_sents=None,
               max_num_train_sents=None, max_src_len=None, max_trg_len=None):
    self.src_file = src_file
    self.trg_file = trg_file
    self.dev_tasks = dev_tasks

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
    self.loss_calculator = loss_calculator or LossCalculator(MLELoss())

    self.sample_train_sents = sample_train_sents
    self.max_num_train_sents = max_num_train_sents
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len

    self.batcher = batcher
    self.logger = BatchLossTracker(self, dev_every, name)

  def load_data(self):
    if self.reload_command is not None:
      self._augmentation_handle = None
      self._augment_data_initial()
    self.src_data, self.trg_data, self.src_batches, self.trg_batches = \
        xnmt.input_reader.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                        self.src_file, self.trg_file,
                                        batcher=self.batcher, sample_sents=self.sample_train_sents,
                                        max_num_sents=self.max_num_train_sents,
                                        max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)

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
          xnmt.input_reader.read_parallel_corpus(self.model.src_reader, self.model.trg_reader,
                                          self.src_file, self.trg_file,
                                          batcher=self.batcher, sample_sents=self.sample_train_sents,
                                          max_num_sents=self.max_num_train_sents,
                                          max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
      # restart data generation
      self._augmentation_handle = Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
    else:
      logger.info('new data set is not ready yet, using data from last epoch.')

  @register_xnmt_event
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
      or self.run_for_epochs is not None and (self.training_state.epoch_num > self.run_for_epochs \
                                              or (self.training_state.epoch_num == self.run_for_epochs and
                                                  self.training_state.steps_into_epoch >= self.cur_num_minibatches() - 1))

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
    Shifts internal state to the next epoch, including batch re-packing and shuffling.
    """
    if self.reload_command is not None:
      self._augment_data_next_epoch()
    self.training_state.epoch_seed = random.randint(1,2147483647)
    random.seed(self.training_state.epoch_seed)
    np.random.seed(self.training_state.epoch_seed)
    self.src_batches, self.trg_batches = \
      self.batcher.pack(self.src_data, self.trg_data)
    self.training_state.epoch_num += 1
    self.training_state.steps_into_epoch = 0
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
        yield src, trg
        self.training_state.steps_into_epoch += 1

  def training_step(self, src, trg):
    """
    Performs forward pass, backward pass, parameter update for the given minibatch
    """
    trg_word_counts = self.logger.count_trg_words(trg)

    loss_builder = LossBuilder()
    standard_loss = self.model.calc_loss(src, trg, self.loss_calculator)
    additional_loss = self.model.calc_additional_loss(src, trg, standard_loss, trg_word_counts)
    loss_builder.add_loss("standard_loss", standard_loss)
    loss_builder.add_loss("additional_loss", additional_loss)
    
    loss_value = loss_builder.compute()
    self.logger.update_epoch_loss(src, trg, loss_builder.get_loss_stats())
    self.logger.report_train_process()

    return loss_value

  def checkpoint_needed(self):
    return self.logger.should_report_dev()

  def checkpoint(self, control_learning_schedule=True):
    """
    Performs a dev checkpoint

    Args:
      control_learning_schedule: If False, only evaluate dev data.
                                      If True, also perform model saving, LR decay etc. if needed.
    Returns:
      True if the model needs saving, False otherwise
    """
    ret = False
    self.logger.new_dev()

    # Perform evaluation
    if self.dev_tasks and len(self.dev_tasks) > 0:
      dev_scores = []
      for dev_task in self.dev_tasks:
        dev_score, dev_word_cnt = dev_task.eval()
        if type(dev_score) == list:
          dev_scores.extend(dev_score)
        else:
          dev_scores.append(dev_score)
      # TODO: This is passing "1" for the number of words, as this is not implemented yet
      self.logger.set_dev_score(dev_word_cnt, dev_scores[0])
      for dev_score in dev_scores[1:]:
        self.logger.report_auxiliary_score(dev_score)

    # Control the learning schedule
    if control_learning_schedule:
      logger.info("> Checkpoint")
      # Write out the model if it's the best one
      if self.logger.report_dev_and_check_model():
        ret = True
        self.training_state.cur_attempt = 0
      else:
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
                ParamManager.param_col.revert_to_best_model()

    return ret

class TrainingState(object):
  """
  This holds the state of the training loop.
  """
  def __init__(self):
    self.num_times_lr_decayed = 0
    self.cur_attempt = 0
    self.epoch_num = 0
    self.steps_into_epoch = 0
    # used to pack and shuffle minibatches; storing helps resuming crashed trainings
    self.epoch_seed = random.randint(1,2147483647)
