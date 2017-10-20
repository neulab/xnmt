from __future__ import division, generators

import sys
import math
import time
import xnmt.loss

class LossTracker(object):
  """
  A template class to track training process and generate report.
  """

  REPORT_TEMPLATE           = 'Epoch %.4f: {}_loss/word=%.3f (words=%d, words/sec=%.2f, time=%s)'
  REPORT_TEMPLATE_DEV       = '  Epoch %.4f dev %s (words=%d, words/sec=%.2f, time=%s)'
  REPORT_TEMPLATE_DEV_AUX   = '  Epoch %.4f dev [auxiliary] %s'

  def __init__(self, eval_every, total_train_sent):
    self.eval_train_every = 1000
    self.eval_dev_every = eval_every
    self.total_train_sent = total_train_sent

    self.epoch_num = 0

    self.epoch_loss = xnmt.loss.LossBuilder()
    self.epoch_words = 0
    self.sent_num = 0
    self.sent_num_not_report_train = 0
    self.sent_num_not_report_dev = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.best_dev_score = None
    self.dev_words = 0

    self.last_report_words = 0
    self.start_time = time.time()
    self.last_report_train_time = self.start_time
    self.dev_start_time = self.start_time

  def new_epoch(self):
    """
    Clear epoch-wise counters for starting a new training epoch.
    """
    self.epoch_loss = xnmt.loss.LossBuilder()
    self.epoch_words = 0
    self.epoch_num += 1
    self.sent_num = 0
    self.sent_num_not_report_train = 0
    self.sent_num_not_report_dev = 0
    self.last_report_words = 0

  def update_epoch_loss(self, src, trg, loss):
    """
    Update epoch-wise counters for each iteration.
    """
    batch_sent_num = self.count_sent_num(src)
    self.sent_num += batch_sent_num
    self.sent_num_not_report_train += batch_sent_num
    self.sent_num_not_report_dev += batch_sent_num
    self.epoch_words += self.count_trg_words(trg)
    self.epoch_loss += loss

  def format_time(self, seconds):
    return "{}-{}".format(int(seconds) // 86400,
                          time.strftime("%H:%M:%S", time.gmtime(seconds)))

  def report_train_process(self):
    """
    Print training report if eval_train_every sents have been evaluated.
    :return: True if the training process is reported
    """
    print_report = self.sent_num_not_report_train >= self.eval_train_every \
                   or self.sent_num == self.total_train_sent

    if print_report:
      self.sent_num_not_report_train = self.sent_num_not_report_train % self.eval_train_every
      self.fractional_epoch = (self.epoch_num - 1) + self.sent_num / self.total_train_sent
      this_report_time = time.time()
      print(LossTracker.REPORT_TEMPLATE.format('train') % (
        self.fractional_epoch, self.epoch_loss.sum() / self.epoch_words,
        self.epoch_words,
        (self.epoch_words - self.last_report_words) / (this_report_time - self.last_report_train_time),
        self.format_time(time.time() - self.start_time)))

      if len(self.epoch_loss) > 1:
        for loss_name, loss_values in self.epoch_loss:
          print("- %s %5.3f" % (loss_name, loss_values / self.epoch_words))

      self.last_report_words = self.epoch_words
      self.last_report_train_time = this_report_time

      return print_report

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
      return self.sent_num_not_report_dev >= self.eval_dev_every or (self.sent_num == self.total_train_sent)
    else:
      return self.sent_num_not_report_dev >= self.total_train_sent

  def report_dev_and_check_model(self, model_file):
    """
    Print dev testing report and check whether the dev loss is the best seen so far.
    :return: True if the dev loss is the best and required save operations
    """
    this_report_time = time.time()
    sent_num = self.eval_dev_every if self.eval_dev_every != 0 else self.total_train_sent
    self.sent_num_not_report_dev = self.sent_num_not_report_dev % sent_num
    self.fractional_epoch = (self.epoch_num - 1) + self.sent_num / self.total_train_sent
    print(LossTracker.REPORT_TEMPLATE_DEV % (
        self.fractional_epoch,
        self.dev_score,
        self.dev_words,
        self.dev_words / (this_report_time - self.dev_start_time),
        self.format_time(this_report_time - self.start_time)))

    save_model = self.dev_score.better_than(self.best_dev_score)
    if save_model:
      self.best_dev_score = self.dev_score
      print('  Epoch %.4f: best dev score, writing model to %s' % (self.fractional_epoch, model_file))

    self.last_report_train_time = time.time()
    return save_model

  def report_auxiliary_score(self, score):
    print(LossTracker.REPORT_TEMPLATE_DEV_AUX % (self.fractional_epoch, score))

  def count_trg_words(self, trg_words):
    """
    Method for counting number of trg words.
    """
    raise NotImplementedError('count_trg_words must be implemented in LossTracker subclasses')

  def count_sent_num(self, obj):
    """
    Method for counting number of sents.
    """
    raise NotImplementedError('count_trg_words must be implemented in LossTracker subclasses')

  def clear_counters(self):
    self.sent_num = 0
    self.sent_num_not_report_dev = 0
    self.sent_num_not_report_train = 0

  def report_loss(self):
    pass


class BatchLossTracker(LossTracker):
  """
  A class to track training process and generate report for minibatch mode.
  """

  def count_trg_words(self, trg_words):
    return sum((1 if type(x) == int else len(x)) for x in trg_words)

  def count_sent_num(self, obj):
    return len(obj)
