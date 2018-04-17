import time

import xnmt.loss
from xnmt.vocab import Vocab
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt import logger, yaml_logger
from xnmt.util import format_time

class TrainLossTracker(object):

  REPORT_TEMPLATE = 'Epoch {epoch:.4f}: {data}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec:.2f}, time={time})'
  EVAL_TRAIN_EVERY = 1000

  @register_xnmt_handler
  def __init__(self, training_task, name=None):
    self.training_task = training_task

    self.epoch_loss = xnmt.loss.LossScalarBuilder()
    self.epoch_words = 0
    self.sent_num_not_report_train = 0

    self.last_report_words = 0

    self.start_time = time.time()
    self.last_report_train_time = self.start_time
    self.name = name

  @handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    """
    Clear epoch-wise counters for starting a new training epoch.
    """
    if training_task is self.training_task:
      self.total_train_sent = num_sents
      self.epoch_loss.zero()
      self.epoch_words = 0
      self.sent_num_not_report_train = 0
      self.last_report_words = 0
      self.last_report_train_time = time.time()

  def update_epoch_loss(self, src, trg, loss):
    """
    Update epoch-wise counters for each iteration.
    """
    batch_sent_num = len(src)
    self.sent_num_not_report_train += batch_sent_num
    self.epoch_words += self.count_trg_words(trg)
    self.epoch_loss += loss

  def log_readable_and_structured(self, template, args):
    if self.name: args["task_name"] = self.name
    logger.info(template.format(**args), extra=args)
    yaml_logger.info(args)

  def report_train_process(self):
    """
    Print training report if EVAL_TRAIN_EVERY sents have been evaluated.

    Return:
      True if the training process is reported
    """
    print_report = self.sent_num_not_report_train >= TrainLossTracker.EVAL_TRAIN_EVERY \
                   or self.training_task.training_state.sents_into_epoch == self.total_train_sent

    if print_report:
      self.sent_num_not_report_train = self.sent_num_not_report_train % TrainLossTracker.EVAL_TRAIN_EVERY
      fractional_epoch = (self.training_task.training_state.epoch_num - 1) + self.training_task.training_state.sents_into_epoch / self.total_train_sent
      this_report_time = time.time()
      self.log_readable_and_structured(TrainLossTracker.REPORT_TEMPLATE,
                                       {"key": "train_loss", "data" : "train",
                                        "epoch" : fractional_epoch,
                                        "loss" : self.epoch_loss.sum() / self.epoch_words,
                                        "words" : self.epoch_words,
                                        "words_per_sec" : (self.epoch_words - self.last_report_words) / (this_report_time - self.last_report_train_time),
                                        "time" : format_time(time.time() - self.start_time)})

      if len(self.epoch_loss) > 1:
        for loss_name, loss_values in self.epoch_loss.items():
          self.log_readable_and_structured("- {loss_name} {loss:5.6f}",
                                           {"key":"additional_train_loss",
                                            "loss_name" : loss_name,
                                            "loss" : loss_values / self.epoch_words})

      self.last_report_words = self.epoch_words
      self.last_report_train_time = this_report_time

      return print_report

  def count_trg_words(self, trg_words):
    trg_cnt = 0
    for x in trg_words:
      if type(x) == int:
        trg_cnt += 1 if x != Vocab.ES else 0
      else:
        trg_cnt += sum([1 if y != Vocab.ES else 0 for y in x])
    return trg_cnt

class DevLossTracker(object):

  REPORT_TEMPLATE_DEV       = 'Epoch {epoch:.4f} dev {score} (words={words}, words/sec={words_per_sec:.2f}, time={time})'
  REPORT_TEMPLATE_DEV_AUX   = 'Epoch {epoch:.4f} dev auxiliary {score}'

  @register_xnmt_handler
  def __init__(self, training_task, eval_every, name=None):
    self.training_task = training_task
    self.eval_dev_every = eval_every

    self.sent_num_not_report_dev = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.best_dev_score = None
    self.dev_words = 0

    self.start_time = time.time()
    self.dev_start_time = self.start_time
    self.name = name

  @handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    """
    Clear epoch-wise counters for starting a new training epoch.
    """
    if training_task is self.training_task:
      self.total_train_sent = num_sents

  def update_epoch_loss(self, src):
    """
    Update epoch-wise counters for each iteration.
    """
    batch_sent_num = len(src)
    self.sent_num_not_report_dev += batch_sent_num

  def log_readable_and_structured(self, template, args):
    if self.name: args["task_name"] = self.name
    logger.info(template.format(**args), extra=args)
    yaml_logger.info(args)

  def new_dev(self):
    """
    Clear dev counters for starting a new dev testing.
    """
    self.dev_start_time = time.time()

  def set_dev_score(self, dev_words, dev_score):
    """
    Update dev counters for each iteration.
    """
    self.dev_score = dev_score
    self.dev_words = dev_words

  def should_report_dev(self):
    if self.eval_dev_every > 0:
      return self.sent_num_not_report_dev >= self.eval_dev_every
    else:
      return self.sent_num_not_report_dev >= self.total_train_sent

  def report_dev_and_check_model(self):
    """
    Print dev testing report and check whether the dev loss is the best seen so far.

    Return:
      True if the dev loss is the best and required save operations
    """
    this_report_time = time.time()
    sent_num = self.eval_dev_every if self.eval_dev_every != 0 else self.total_train_sent
    self.sent_num_not_report_dev = self.sent_num_not_report_dev % sent_num
    self.fractional_epoch = (self.training_task.training_state.epoch_num - 1) + self.training_task.training_state.sents_into_epoch / self.total_train_sent
    self.log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV,
                                     {"key" : "dev_loss",
                                      "epoch" : self.fractional_epoch,
                                      "score" : self.dev_score,
                                      "words" : self.dev_words,
                                      "words_per_sec" :  self.dev_words / (this_report_time - self.dev_start_time),
                                      "time" : format_time(this_report_time - self.start_time)
                                      })

    save_model = True
    if self.best_dev_score is not None:
      save_model = self.dev_score.better_than(self.best_dev_score)
    if save_model:
      self.best_dev_score = self.dev_score
      logger.info(f"Epoch {self.fractional_epoch:.4f}: best dev score, writing out model")
    return save_model

  def report_auxiliary_score(self, score):
    self.log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV_AUX,
                                     {"key": "auxiliary_score",
                                      "epoch" : self.fractional_epoch,
                                      "score" : score})

