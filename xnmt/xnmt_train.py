# coding: utf-8
from __future__ import division

import argparse
import math
import sys
import dynet as dy
import six
from embedder import *
from attender import *
from input import *
from encoder import *
from specialized_encoders import *
from decoder import *
from translator import *
from retriever import *
from serialize_container import *
from training_corpus import *
from loss_tracker import *
from preproc import SentenceFilterer
from options import Option, OptionParser, general_options
from loss import LossBuilder
import model_globals
import serializer
import xnmt_decode
import xnmt_evaluate
from evaluator import LossScore
from tee import Tee
'''
This will be the main class to perform training.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("dev_every", int, default_value=0, force_flag=True, help_str="dev checkpoints every n sentences (0 for only after epoch)"),
  Option("batch_size", int, default_value=32, force_flag=True),
  Option("batch_strategy", default_value="src"),
  Option("training_corpus"),
  Option("corpus_parser"),
#  Option("train_filters", list, required=False, help_str="Specify filtering criteria for the training data"),
#  Option("dev_filters", list, required=False, help_str="Specify filtering criteria for the development data"),
  Option("model_file"),
  Option("save_num_checkpoints", int, default_value=1, help_str="Save recent n best checkpoints"),
  Option("pretrained_model_file", default_value="", help_str="Path of pre-trained model file"),
  Option("src_format", default_value="text", help_str="Format of input data: text/contvec"),
  Option("default_layer_dim", int, default_value=512, help_str="Default size to use for layers if not otherwise overridden"),
  Option("trainer", default_value="sgd"),
  Option("learning_rate", float, default_value=0.1),
  Option("momentum", float, default_value = 0.9),
  Option("lr_decay", float, default_value=1.0),
  Option("lr_decay_times", int, default_value=3, help_str="Early stopping after decaying learning rate a certain number of times"),
  Option("attempts_before_lr_decay", int, default_value=1, help_str="apply LR decay after dev scores haven't improved over this many checkpoints"),
  Option("dev_metrics", default_value="", help_str="Comma-separated list of evaluation metrics (bleu/wer/cer)"),
  Option("schedule_metric", default_value="loss", help_str="determine learning schedule based on this dev_metric (loss/bleu/wer/cer)"),
  Option("restart_trainer", bool, default_value=False, help_str="Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)"),
  Option("reload_between_epochs", bool, default_value=False, help_str="Reload train data between epochs (useful when sampling from train data, or with noisy input data via an external tool"),
  Option("dropout", float, default_value=0.0),
  Option("model", dict, default_value={}),
]

class XnmtTrainer(object):
  def __init__(self, args, output=None):
    dy.renew_cg()

    self.args = args
    self.output = output
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection(self.args.model_file, self.args.save_num_checkpoints)

    self.trainer = self.dynet_trainer_for_args(args)

    if args.lr_decay > 1.0 or args.lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.num_times_lr_decayed = 0
    self.early_stopping_reached = False
    self.cur_attempt = 0

    self.evaluators = [s.lower() for s in self.args.dev_metrics.split(",") if s.strip()!=""]
    if self.args.schedule_metric.lower() not in self.evaluators:
              self.evaluators.append(self.args.schedule_metric.lower())
    if "loss" not in self.evaluators: self.evaluators.append("loss")

    # Initialize the serializer
    self.model_serializer = serializer.YamlSerializer()

    if self.args.pretrained_model_file:
      self.load_corpus_and_model()
    else:
      self.create_corpus_and_model()

    # single mode
    if not self.is_batch_mode():
      print('Start training in non-minibatch mode...')
      self.logger = NonBatchLossTracker(args.dev_every, self.total_train_sent)
      self.train_src, self.train_trg = \
          self.training_corpus.train_src_data, self.training_corpus.train_trg_data
      self.dev_src, self.dev_trg = \
          self.training_corpus.dev_src_data, self.training_corpus.dev_trg_data

    # minibatch mode
    else:
      print('Start training in minibatch mode...')
      self.batcher = Batcher.from_spec(args.batch_strategy, args.batch_size)
      if args.src_format == "contvec":
        self.batcher.pad_token = np.zeros(self.model.src_embedder.emb_dim)
      self.pack_batches()
      self.logger = BatchLossTracker(args.dev_every, self.total_train_sent)
  def is_batch_mode(self):
    return not (self.args.batch_size is None or self.args.batch_size == 1 or self.args.batch_strategy.lower() == 'none')
  def pack_batches(self):
    self.train_src, self.train_trg = \
      self.batcher.pack(self.training_corpus.train_src_data, self.training_corpus.train_trg_data)
    self.dev_src, self.dev_trg = \
      self.batcher.pack(self.training_corpus.dev_src_data, self.training_corpus.dev_trg_data)

  def dynet_trainer_for_args(self, args):
    if args.trainer.lower() == "sgd":
      trainer = dy.SimpleSGDTrainer(model_globals.dynet_param_collection.param_col, args.learning_rate)
    elif args.trainer.lower() == "adam":
      trainer = dy.AdamTrainer(model_globals.dynet_param_collection.param_col, alpha = args.learning_rate)
    elif args.trainer.lower() == "msgd":
      trainer = dy.MomentumSGDTrainer(model_globals.dynet_param_collection.param_col, args.learning_rate, mom = args.momentum)
    else:
      raise RuntimeError("Unknown trainer {}".format(args.trainer))
    return trainer

  def create_corpus_and_model(self):
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus)
    self.corpus_parser = self.model_serializer.initialize_object(self.args.corpus_parser)
    self.corpus_parser.read_training_corpus(self.training_corpus)
    self.total_train_sent = len(self.training_corpus.train_src_data)
    context={"corpus_parser" : self.corpus_parser, "training_corpus":self.training_corpus}
    model_globals.model_globals["default_layer_dim"] = self.args.default_layer_dim
    model_globals.model_globals["dropout"] = self.args.dropout
    self.model = self.model_serializer.initialize_object(self.args.model, context)

  def load_corpus_and_model(self):
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus)
    corpus_parser, model, my_model_globals = self.model_serializer.load_from_file(self.args.pretrained_model_file, model_globals.dynet_param_collection)
    self.corpus_parser = self.model_serializer.initialize_object(corpus_parser)
    self.corpus_parser.read_training_corpus(self.training_corpus)
    model_globals.model_globals = my_model_globals
    self.total_train_sent = len(self.training_corpus.train_src_data)
    context={"corpus_parser" : self.corpus_parser, "training_corpus":self.training_corpus}
    self.model = self.model_serializer.initialize_object(model, context)
    model_globals.dynet_param_collection.load_from_data_file(self.args.pretrained_model_file + '.data')


#  def read_data(self):
#    train_filters = SentenceFilterer.from_spec(self.args.train_filters)
#    self.train_src, self.train_trg = \
#        self.filter_sents(self.src_reader.read_file(self.args.train_src, max_num=self.args.max_num_train_sents),
#                          self.trg_reader.read_file(self.args.train_trg, max_num=self.args.max_num_train_sents),
#                          train_filters)
#    assert len(self.train_src) == len(self.train_trg)
#    self.total_train_sent = len(self.train_src)
#    if self.args.dev_every == None:
#      self.args.dev_every = self.total_train_sent
#
#    self.src_reader.freeze()
#    self.trg_reader.freeze()
#
#    dev_filters = SentenceFilterer.from_spec(self.args.dev_filters)
#    self.dev_src, self.dev_trg = \
#        self.filter_sents(self.src_reader.read_file(self.args.dev_src),
#                          self.trg_reader.read_file(self.args.dev_trg),
#                          dev_filters)
#    assert len(self.dev_src) == len(self.dev_trg)

#  def filter_sents(self, src_sents, trg_sents, my_filters):
#    if len(my_filters) == 0:
#      return src_sents, trg_sents
#    filtered_src_sents, filtered_trg_sents = [], []
#    for src_sent, trg_sent in zip(src_sents, trg_sents):
#      if all([my_filter.keep((src_sent,trg_sent)) for my_filter in my_filters]):
#        filtered_src_sents.append(src_sent)
#        filtered_trg_sents.append(trg_sent)
#    print("> removed %s out of %s sentences that didn't pass filters." % (len(src_sents)-len(filtered_src_sents),len(src_sents)))
#    return filtered_src_sents, filtered_trg_sents

  def run_epoch(self):
    self.logger.new_epoch()

    if self.logger.epoch_num > 1:
      if self.args.reload_between_epochs:
        print("Reloading training data..")
        self.corpus_parser.read_training_corpus(self.training_corpus)
        if self.is_batch_mode():
          self.pack_batches()
      elif self.is_batch_mode() and self.batcher.is_random():
        self.pack_batches()

    self.model.set_train(True)
    order = list(range(0, len(self.train_src)))
    np.random.shuffle(order)
    for batch_num in order:
      src, trg = self.train_src[batch_num], self.train_trg[batch_num]

      # Loss calculation
      dy.renew_cg()
      loss = LossBuilder()
      loss.add_node(self.model.calc_loss, [src, trg])
      loss.add_node(self.model.calc_reinforce_loss, [-loss[-1]])

      # Log the loss sum
      self.logger.update_epoch_loss(src, trg, loss)

      loss.compute().backward()
      self.trainer.update()

      # Devel reporting
      self.logger.report_train_process()
      if self.logger.should_report_dev():
        self.model.set_train(False)
        self.logger.new_dev()
        trg_words_cnt, loss_score = self.compute_dev_loss()
        schedule_metric = self.args.schedule_metric.lower()

        eval_scores = {"loss" : loss_score}
        if filter(lambda e: e!="loss", self.evaluators):
          self.decode_args.src_file = self.training_corpus.dev_src
          self.decode_args.candidate_id_file = self.training_corpus.dev_id_file
          out_file = self.args.model_file + ".dev_hyp"
          out_file_ref = self.args.model_file + ".dev_ref"
          self.decode_args.trg_file = out_file
          xnmt_decode.xnmt_decode(self.decode_args, model_elements=(self.corpus_parser, self.model))
          output_processor = xnmt_decode.output_processor_for_spec(self.decode_args.post_process)
          processed = []
          with io.open(self.training_corpus.dev_trg, encoding='utf-8') as fin:
            for line in fin:
              processed.append(output_processor.words_to_string(line.strip().split()) + u"\n")
          with io.open(out_file_ref, 'wt', encoding='utf-8') as fout:
            for line in processed:
              fout.write(line)
          self.evaluate_args.hyp_file = out_file
          self.evaluate_args.ref_file = out_file_ref
          for evaluator in self.evaluators:
            if evaluator=="loss": continue
            self.evaluate_args.evaluator = evaluator
            eval_score = xnmt_evaluate.xnmt_evaluate(self.evaluate_args)
            eval_scores[evaluator] = eval_score
        if schedule_metric == "loss":
          self.logger.set_dev_score(trg_words_cnt, loss_score)
        else:
          self.logger.set_dev_score(trg_words_cnt, eval_scores[schedule_metric])

        print("> Checkpoint")
        # print previously computed metrics
        for metric in self.evaluators:
          if metric != schedule_metric:
            self.logger.report_auxiliary_score(eval_scores[metric])
        # Write out the model if it's the best one
        if self.logger.report_dev_and_check_model(self.args.model_file):
          self.model_serializer.save_to_file(self.args.model_file,
                                             SerializeContainer(self.corpus_parser, self.model, model_globals.model_globals),
                                             model_globals.dynet_param_collection)
          self.cur_attempt = 0
        else:
          # otherwise: learning rate decay / early stopping
          self.cur_attempt += 1
          if self.args.lr_decay < 1.0 and self.cur_attempt >= self.args.attempts_before_lr_decay:
            self.num_times_lr_decayed += 1
            if self.num_times_lr_decayed > self.args.lr_decay_times:
              print('  Early stopping')
              self.early_stopping_reached = True
            else:
              self.trainer.learning_rate *= self.args.lr_decay
              print('  new learning rate: %s' % self.trainer.learning_rate)
              if self.args.restart_trainer:
                print('  restarting trainer and reverting learned weights to best checkpoint..')
                self.trainer = self.dynet_trainer_for_args(self.args)
                model_globals.dynet_param_collection.revert_to_best_model()


        self.model.set_train(True)
        self.model.new_epoch()


  def compute_dev_loss(self):
    loss = LossBuilder()
    trg_words_cnt = 0
    for src, trg in zip(self.dev_src, self.dev_trg):
      dy.renew_cg()
      loss.add_node(self.model.calc_loss, [src, trg])
      trg_words_cnt += self.logger.count_trg_words(trg)
      loss.compute()
    return trg_words_cnt, LossScore(loss.sum() / trg_words_cnt)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_task("train", general_options + options)
  args = parser.args_from_command_line("train", sys.argv[1:])
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while not xnmt_trainer.early_stopping_reached:
    xnmt_trainer.run_epoch()
