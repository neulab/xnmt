from __future__ import division, generators

import sys
import math
import time
import json
import codecs

class LossTracker:
    """
    A template class to track training process and generate report.
    """

    REPORT_TEMPLATE = 'Epoch %.4f: {}_ppl=%.4f (words=%d, words/sec=%.2f, time=%s)'

    def __init__(self, eval_every, total_train_sent):
        self.eval_every = eval_every
        self.total_train_sent = total_train_sent

        self.epoch_num = 0

        self.epoch_loss = 0.0
        self.epoch_words = 0
        self.sent_num = 0
        self.sent_num_not_report = 0
        self.fractional_epoch = 0

        self.dev_loss = 0.0
        self.best_dev_loss = sys.float_info.max
        self.dev_words = 0

        self.last_report_words = 0
        self.start_time = time.time()
        self.last_report_train_time = self.start_time
        self.dev_start_time = self.start_time

    def new_epoch(self):
        """
        Clear epoch-wise counters for starting a new training epoch.
        """
        self.epoch_loss = 0.0
        self.epoch_words = 0
        self.epoch_num += 1
        self.sent_num = 0
        self.sent_num_not_report = 0

    def update_epoch_loss(self, src, tgt, loss):
        """
        Update epoch-wise counters for each iteration.
        """
        batch_sent_num = self.count_sent_num(src)
        self.sent_num += batch_sent_num
        self.sent_num_not_report += batch_sent_num
        self.epoch_words += self.count_tgt_words(tgt)
        self.epoch_loss += loss
    
    def format_time(self, seconds):
        return "{}-{}".format(int(seconds) // 86400, 
                              time.strftime("%H:%M:%S", time.gmtime(seconds)),
                             )

    def report_train_process(self):
        """
        Print training report if eval_every sentences have been evaluated.
        :return: True if the training process is reported
        """
        print_report = (self.sent_num_not_report >= self.eval_every) or (self.sent_num == self.total_train_sent)

        if print_report:
            while self.sent_num_not_report >= self.eval_every:
                self.sent_num_not_report -= self.eval_every
            self.fractional_epoch = (self.epoch_num - 1) + self.sent_num / self.total_train_sent

            this_report_time = time.time()
            print(LossTracker.REPORT_TEMPLATE.format('train') % (
                self.fractional_epoch, math.exp(self.epoch_loss / self.epoch_words),
                self.epoch_words,
                (self.epoch_words - self.last_report_words) / (this_report_time - self.last_report_train_time),
                self.format_time(time.time() - self.start_time)))
            self.last_report_words = self.epoch_words
            self.last_report_train_time = this_report_time

        return print_report

    def new_dev(self):
        """
        Clear dev counters for starting a new dev testing.
        """
        self.dev_loss = 0.0
        self.dev_words = 0
        self.dev_start_time = time.time()

    def update_dev_loss(self, tgt, loss):
        """
        Update dev counters for each iteration.
        """
        self.dev_loss += loss
        self.dev_words += self.count_tgt_words(tgt)

    def report_dev_and_check_model(self, model_file):
        """
        Print dev testing report and check whether the dev loss is the best seen so far.
        :return: True if the dev loss is the best and required save operations
        """
        this_report_time = time.time()
        print(LossTracker.REPORT_TEMPLATE.format('test') % (
            self.fractional_epoch,
            math.exp(self.dev_loss / self.dev_words),
            self.dev_words,
            self.dev_words / (this_report_time - self.dev_start_time),
            self.format_time(this_report_time - self.start_time)))

        save_model = self.dev_loss < self.best_dev_loss
        if save_model:
            self.best_dev_loss = self.dev_loss
            print('Epoch %.4f: best dev loss, writing model to %s' % (self.fractional_epoch, model_file))

        self.last_report_train_time = time.time()
        return save_model

    def count_tgt_words(self, tgt_words):
        """
        Method for counting number of target words.
        """
        raise NotImplementedError('count_tgt_words must be implemented in LossTracker subclasses')

    def count_sent_num(self, obj):
        """
        Method for counting number of sentences.
        """
        raise NotImplementedError('count_tgt_words must be implemented in LossTracker subclasses')

    def clear_counters(self):
        self.sent_num = 0
        self.sent_num_not_report = 0

    def report_ppl(self):
        pass


class BatchLossTracker(LossTracker):
    """
    A class to track training process and generate report for minibatch mode.
    """

    def count_tgt_words(self, tgt_words):
        return sum(len(x) for x in tgt_words)

    def count_sent_num(self, obj):
        return len(obj)


class NonBatchLossTracker(LossTracker):
    """
    A class to track training process and generate report for non-minibatch mode.
    """

    def count_tgt_words(self, tgt_words):
        return len(tgt_words)

    def count_sent_num(self, obj):
        return 1
