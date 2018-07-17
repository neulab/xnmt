from typing import Union
import time

import numpy as np

from xnmt import batcher, input, loss, vocab, events, util

class AccumTimeTracker(object):
  def __init__(self):
    self.start_time = None
    self.accum_time = 0.0

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, *args):
    self.accum_time += time.time() - self.start_time

  def get_and_reset(self):
    ret = self.accum_time
    self.accum_time = 0.0
    return ret

class TrainLossTracker(object):

  REPORT_TEMPLATE_SPEED = 'Epoch {epoch:.4f}: {data}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec:.2f}, time={time})'
  REPORT_TEMPLATE = 'Epoch {epoch:.4f}: {data}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec}, time={time})'
  REPORT_TEMPLATE_ADDITIONAL = '- {loss_name} {loss:5.6f}'
  REPORT_EVERY = 1000

  @events.register_xnmt_handler
  def __init__(self, training_task):
    self.training_task = training_task

    self.epoch_loss = loss.FactoredLossVal()
    self.epoch_words = 0
    self.last_report_sents_into_epoch = 0
    self.last_report_sents_since_start = 0

    self.last_report_words = 0

    self.time_tracker = AccumTimeTracker()
    self.start_time = time.time()
    self.name = self.training_task.name

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    if training_task is self.training_task:
      self.epoch_loss.clear()
      self.epoch_words = 0
      self.last_report_sents_since_start = 0
      self.last_report_words = 0

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
      accum_time = self.time_tracker.get_and_reset()
      util.log_readable_and_structured(
        TrainLossTracker.REPORT_TEMPLATE_SPEED if accum_time else TrainLossTracker.REPORT_TEMPLATE,
        {"key": "train_loss", "data": "train",
         "epoch": fractional_epoch,
         "loss": self.epoch_loss.sum_factors() / self.epoch_words,
         "words": self.epoch_words,
         "words_per_sec": (self.epoch_words - self.last_report_words) / (
           accum_time) if accum_time else "-",
         "time": util.format_time(time.time() - self.start_time)},
        task_name=self.name)

      if len(self.epoch_loss) > 1:
        for loss_name, loss_values in self.epoch_loss.items():
          util.log_readable_and_structured(TrainLossTracker.REPORT_TEMPLATE_ADDITIONAL,
                                           {"key": "additional_train_loss",
                                            "loss_name": loss_name,
                                            "loss": loss_values / self.epoch_words},
                                           task_name=self.name)

      self.last_report_words = self.epoch_words
      self.last_report_sents_since_start = self.training_task.training_state.sents_since_start

  def count_trg_words(self, trg_words: Union[input.Input, batcher.Batch]) -> int:
    if isinstance(trg_words, batcher.Batch):
      return sum(inp.len_unpadded() for inp in trg_words)
    else:
      return trg_words.len_unpadded()

class DevLossTracker(object):

  REPORT_TEMPLATE_DEV         = 'Epoch {epoch:.4f} dev {score} (time={time})'
  REPORT_TEMPLATE_DEV_AUX     = '             dev auxiliary {score}'
  REPORT_TEMPLATE_TIME_NEEDED = '             checkpoint took {time_needed}'

  def __init__(self, training_task, eval_every, name=None):
    self.training_task = training_task
    self.eval_dev_every = eval_every

    self.last_report_sents_since_start = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.aux_scores = []

    self.start_time = time.time()
    self.name = name
    self.time_tracker = AccumTimeTracker()

  def set_dev_score(self, dev_score):
    self.dev_score = dev_score

  def add_aux_score(self, score):
    self.aux_scores.append(score)

  def should_report_dev(self):
    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    if self.eval_dev_every > 0:
      return sent_num_not_report >= self.eval_dev_every
    else:
      return sent_num_not_report >= self.training_task.cur_num_sentences()

  def report(self):
    this_report_time = time.time()
    self.last_report_sents_since_start = self.training_task.training_state.sents_since_start
    self.fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                            + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
    dev_time = self.time_tracker.get_and_reset()
    util.log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV,
                                     {"key": "dev_loss",
                                      "epoch": self.fractional_epoch,
                                      "score": self.dev_score,
                                      "time": util.format_time(this_report_time - self.start_time)
                                      },
                                     task_name=self.name)
    for score in self.aux_scores:
      util.log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_DEV_AUX,
                                       {"key": "auxiliary_score", "epoch": self.fractional_epoch, "score": score},
                                       task_name=self.name)
    util.log_readable_and_structured(DevLossTracker.REPORT_TEMPLATE_TIME_NEEDED,
                                     {"key": "dev_time_needed", "epoch": self.fractional_epoch,
                                      "time_needed": util.format_time(dev_time)},
                                     task_name=self.name)
    self.aux_scores = []

