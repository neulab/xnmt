import time

import xnmt.loss
from xnmt.vocab import Vocab
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt import logger, yaml_logger
from xnmt.util import format_time, log_readable_and_structured

class TrainLossTracker(object):

  REPORT_TEMPLATE = 'Epoch {epoch:.4f}: {data}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec:.2f}, time={time})'
  REPORT_TEMPLATE_ADDITIONAL = '- {loss_name} {loss:5.6f}'
  REPORT_EVERY = 1000

  @register_xnmt_handler
  def __init__(self, training_task, name=None):
    self.training_task = training_task

    self.epoch_loss = xnmt.loss.LossScalarBuilder()
    self.epoch_words = 0
    self.last_report_sents_into_epoch = 0
    self.last_report_sents_since_start = 0

    self.last_report_words = 0

    self.accumulated_time = 0
    self.name = name

  @handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    """
    Clear epoch-wise counters for starting a new training epoch.
    """
    if training_task is self.training_task:
      self.epoch_loss.zero()
      self.epoch_words = 0
      self.last_report_sents_since_start = 0
      self.last_report_words = 0

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, *args):
    self.accumulated_time += time.time() - self.start_time

  def report(self, trg, loss):
    """
    Accumulate training loss and report every REPORT_EVERY sentences.
    """
    self.epoch_words += self.count_trg_words(trg)
    self.epoch_loss += loss

    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    should_report = sent_num_not_report >= TrainLossTracker.REPORT_EVERY \
                    or self.training_task.training_state.sents_into_epoch == self.training_task.cur_num_sentences()

    if should_report:
      fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                         + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
      log_readable_and_structured(TrainLossTracker.REPORT_TEMPLATE,
                                  {"key": "train_loss", "data": "train",
                                   "epoch": fractional_epoch,
                                   "loss": self.epoch_loss.sum() / self.epoch_words,
                                   "words": self.epoch_words,
                                   "words_per_sec": (self.epoch_words - self.last_report_words) / (
                                             self.accumulated_time),
                                   "time": format_time(time.time() - self.start_time)},
                                  task_name=self.name)

      if len(self.epoch_loss) > 1:
        for loss_name, loss_values in self.epoch_loss.items():
          log_readable_and_structured(TrainLossTracker.REPORT_TEMPLATE_ADDITIONAL,
                                      {"key": "additional_train_loss",
                                       "loss_name": loss_name,
                                       "loss": loss_values / self.epoch_words},
                                      task_name=self.name)

      self.last_report_words = self.epoch_words
      self.accumulated_time = 0

      self.last_report_sents_since_start = self.training_task.training_state.sents_since_start

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

  def __init__(self, training_task, eval_every, name=None):
    self.training_task = training_task
    self.eval_dev_every = eval_every

    self.last_report_sents_since_start = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.best_dev_score = None
    self.dev_words = 0

    self.start_time = time.time()
    self.dev_start_time = self.start_time
    self.name = name

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
    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    if self.eval_dev_every > 0:
      return sent_num_not_report >= self.eval_dev_every
    else:
      return sent_num_not_report >= self.training_task.cur_num_sentences()

  def report_dev_and_check_model(self):
    """
    Print dev testing report and check whether the dev loss is the best seen so far.

    Return:
      True if the dev loss is the best and required save operations
    """
    this_report_time = time.time()
    self.last_report_sents_since_start = self.training_task.training_state.sents_since_start
    self.fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                            + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
    log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV,
                                {"key": "dev_loss",
                                 "epoch": self.fractional_epoch,
                                 "score": self.dev_score,
                                 "words": self.dev_words,
                                 "words_per_sec": self.dev_words / (this_report_time - self.dev_start_time),
                                 "time": format_time(this_report_time - self.start_time)
                                 },
                                task_name=self.name)

    save_model = True
    if self.best_dev_score is not None:
      save_model = self.dev_score.better_than(self.best_dev_score)
    if save_model:
      self.best_dev_score = self.dev_score
      logger.info(f"Epoch {self.fractional_epoch:.4f}: best dev score, writing out model")
    return save_model

  def report_auxiliary_score(self, score):
    log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV_AUX,
                                {"key": "auxiliary_score", "epoch": self.fractional_epoch, "score": score},
                                task_name=self.name)
