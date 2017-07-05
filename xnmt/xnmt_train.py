# coding: utf-8
from __future__ import division

import argparse
import math
import sys
import dynet as dy
from embedder import *
from attender import *
from input import *
from encoder import *
from decoder import *
from translator import *
from retriever import *
from model_params import *
from training_corpus import *
from loss_tracker import *
from serializer import *
from serializer import *
from preproc import SentenceFilterer
from options import Option, OptionParser, general_options
import model_globals
import serializer

'''
This will be the main class to perform training.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("dev_every", int, default_value=10000, force_flag=True),
  Option("batch_size", int, default_value=32, force_flag=True),
  Option("batch_strategy", default_value="src"),
  Option("training_corpus"),
  Option("corpus_parser"),
#  Option("train_filters", list, required=False, help_str="Specify filtering criteria for the training data"),
#  Option("dev_filters", list, required=False, help_str="Specify filtering criteria for the development data"),
  Option("model_file"),
  Option("pretrained_model_file", default_value="", help_str="Path of pre-trained model file"),
  Option("src_format", default_value="text", help_str="Format of input data: text/contvec"),
  Option("default_layer_dim", int, default_value=512, help_str="Default size to use for layers if not otherwise overridden"),
  Option("trainer", default_value="sgd"),
  Option("learning_rate", float, default_value=0.1),
  Option("lr_decay", float, default_value=1.0),
  Option("lr_decay_times", int, default_value=3, help_str="Early stopping after decaying learning rate a certain number of times"),
  Option("dev_metrics", default_value="bleu", help_str="Comma-separated list of evaluation metrics (bleu/wer/cer)"),
  Option("schedule_use_metric", bool, default_value=False, help_str="determine learning schedule based on the first given dev_metric, instead of PPL"),
  Option("restart_trainer", bool, default_value=False, help_str="Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)"),
  Option("dropout", float, default_value=0.0),
  Option("model", dict, default_value={}),  
]

class XnmtTrainer:
  def __init__(self, args):
    dy.renew_cg()

    self.args = args  # save for later
    model_globals.params["model"] = dy.Model()

    self.trainer = self.trainer_for_args(args)
    
    if args.lr_decay > 1.0 or args.lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.learning_scale = 1.0
    self.num_times_lr_decayed = 0
    self.early_stopping_reached = False

    # Initialize the serializer
    self.model_serializer = serializer.YamlSerializer()

    if self.args.pretrained_model_file:
      self.load_corpus_and_model()
    else:
      self.create_corpus_and_model()

    # single mode
    if args.batch_size is None or args.batch_size == 1 or args.batch_strategy.lower() == 'none':
      print('Start training in non-minibatch mode...')
      self.logger = NonBatchLossTracker(args.dev_every, self.total_train_sent)
      self.train_src, self.train_trg = \
          self.training_corpus.train_src_data, self.training_corpus.train_trg_data
      self.dev_src, self.dev_trg = \
          self.training_corpus.dev_src_data, self.training_corpus.dev_trg_data

    # minibatch mode
    else:
      print('Start training in minibatch mode...')
      self.batcher = Batcher.select_batcher(args.batch_strategy)(args.batch_size)
      if args.src_format == "contvec":
        assert self.train_src[0].nparr.shape[1] == self.src_embedder.emb_dim, "input embed dim is different size than expected"
        self.batcher.pad_token = np.zeros(self.src_embedder.emb_dim)
      self.train_src, self.train_trg = \
          self.batcher.pack(self.training_corpus.train_src_data, self.training_corpus.train_trg_data)
      self.dev_src, self.dev_trg = \
          self.batcher.pack(self.training_corpus.dev_src_data, self.training_corpus.dev_trg_data)
      self.logger = BatchLossTracker(args.dev_every, self.total_train_sent)

  def trainer_for_args(self, args):
    if args.trainer.lower() == "sgd":
      trainer = dy.SimpleSGDTrainer(model_globals.get("model"), e0 = args.learning_rate)
    elif args.trainer.lower() == "adam":
      trainer = dy.AdamTrainer(model_globals.get("model"), alpha = args.learning_rate)
    else:
      raise RuntimeError("Unknown trainer {}".format(args.trainer))
    return trainer

  def create_corpus_and_model(self):
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus)
    self.corpus_parser = self.model_serializer.initialize_object(self.args.corpus_parser)
    self.corpus_parser.read_training_corpus(self.training_corpus)
    self.total_train_sent = len(self.training_corpus.train_src_data)
    context={"corpus_parser" : self.corpus_parser, "training_corpus":self.training_corpus}
    model_globals.params["default_layer_dim"] = self.args.default_layer_dim
    model_globals.params["dropout"] = self.args.dropout
    self.model = self.model_serializer.initialize_object(self.args.model, context)
  
  def load_corpus_and_model(self):
    corpus_parser, model, global_params = self.model_serializer.load_from_file(self.args.pretrained_model_file, model_globals.get("model"))
    self.training_corpus = self.model_serializer.initialize_object(self.args.training_corpus)
    self.corpus_parser = self.model_serializer.initialize_object(corpus_parser)
    self.corpus_parser.read_training_corpus(self.training_corpus)
    model_globals.params = global_params
    self.total_train_sent = len(self.training_corpus.train_src_data)
    context={"corpus_parser" : self.corpus_parser, "training_corpus":self.training_corpus}
    self.model = self.model_serializer.initialize_object(model, context)
    
    
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

    self.model.set_train(True)
    for batch_num, (src, trg) in enumerate(zip(self.train_src, self.train_trg)):

      # Loss calculation
      dy.renew_cg()
      loss = self.model.calc_loss(src, trg)
      self.logger.update_epoch_loss(src, trg, loss.value())

      loss.backward()
      self.trainer.update(self.learning_scale)
      

      # Devel reporting
      self.logger.report_train_process()
      if self.logger.should_report_dev():
        self.model.set_train(False)
        self.logger.new_dev()
        for src, trg in zip(self.dev_src, self.dev_trg):
          dy.renew_cg()
          loss = self.model.calc_loss(src, trg).value()
          self.logger.update_dev_loss(trg, loss)

        # Write out the model if it's the best one
        if self.logger.report_dev_and_check_model(self.args.model_file):
          self.model_serializer.save_to_file(self.args.model_file, 
                                             ModelParams(self.corpus_parser, self.model, model_globals.params),
                                             model_globals.get("model"))
        else:
          # otherwise: learning rate decay / early stopping
          if self.args.lr_decay < 1.0:
            self.num_times_lr_decayed += 1
            if self.num_times_lr_decayed > self.args.lr_decay_times:
              print('Early stopping')
              self.early_stopping_reached = True
            else:
              self.learning_scale *= self.args.lr_decay
              print('new learning rate: %s' % (self.learning_scale * self.args.learning_rate))
              if self.args.restart_trainer:
                print('restarting trainer and reverting learned weights to best checkpoint..')
                self.trainer = self.trainer_for_args(self.args)
                self.revert_to_best_model()
                
            
        self.trainer.update_epoch()
        self.model.set_train(True)

    return math.exp(self.logger.epoch_loss / self.logger.epoch_words), \
           math.exp(self.logger.dev_loss / self.logger.dev_words)

  def revert_to_best_model(self):
    try: # dynet v2
      model_globals.get("model").populate(self.args.model_file + '.data')
    except AttributeError: # dynet v1
      model_globals.get("model").load_all(self.args.model_file + '.data')

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_task("train", general_options + options)
  args = parser.args_from_command_line("train", sys.argv[1:])
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while not xnmt_trainer.early_stopping_reached:
    xnmt_trainer.run_epoch()
