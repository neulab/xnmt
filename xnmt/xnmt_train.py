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
from model_params import *
from logger import *
from serializer import *

'''
This will be the main class to perform training.
'''


class XnmtTrainer:
  def __init__(self, args, encoder_builder=BiLSTMEncoder, encoder_layers=2, decoder_builder=dy.LSTMBuilder,
               decoder_layers=2):
    dy.renew_cg()

    self.args = args  # save for later

    self.model = dy.Model()
    if args.trainer == "sgd":
      self.trainer = dy.SimpleSGDTrainer(self.model)
    elif args.trainer == "adam":
      self.trainer = dy.AdamTrainer(self.model)
    else:
      raise RuntimeError("Unkonwn trainer {}".format(args.trainer))

    # Create the model serializer
    self.model_serializer = JSONSerializer()

    # Read in training and dev corpora
    self.input_reader = InputReader.create_input_reader(args.input_format)
    self.output_reader = InputReader.create_input_reader("text")

    self.train_corpus_source = self.input_reader.read_file(args.train_source)
    self.train_corpus_target = self.output_reader.read_file(args.train_target)
    assert len(self.train_corpus_source) == len(self.train_corpus_target)
    total_train_sent = len(self.train_corpus_source)
    if args.eval_every == None:
      args.eval_every = total_train_sent

    self.input_reader.freeze()
    self.output_reader.freeze()

    self.dev_corpus_source = self.input_reader.read_file(args.dev_source)
    self.dev_corpus_target = self.output_reader.read_file(args.dev_target)
    assert len(self.dev_corpus_source) == len(self.dev_corpus_target)

    # Create the translator object and all its subparts
    self.input_word_emb_dim = args.input_word_embed_dim
    self.output_word_emb_dim = args.output_word_emb_dim
    self.output_state_dim = args.output_state_dim
    self.attender_hidden_dim = args.attender_hidden_dim
    self.output_mlp_hidden_dim = args.output_mlp_hidden_dim
    self.encoder_hidden_dim = args.encoder_hidden_dim

    if args.input_format == "text":
      self.input_embedder = SimpleWordEmbedder(len(self.input_reader.vocab), self.input_word_emb_dim, self.model)
    elif args.input_format == "contvec":
      self.input_embedder = FeatVecNoopEmbedder(self.input_word_emb_dim, self.model)
    else:
      raise RuntimeError("Unkonwn input type {}".format(args.input_format))
    self.output_embedder = SimpleWordEmbedder(len(self.output_reader.vocab), self.output_word_emb_dim, self.model)
    self.encoder = encoder_builder(encoder_layers, self.encoder_hidden_dim, self.input_embedder, self.model)
    self.attender = StandardAttender(self.encoder_hidden_dim, self.output_state_dim, self.attender_hidden_dim, self.model)
    self.decoder = MlpSoftmaxDecoder(decoder_layers, self.encoder_hidden_dim, self.output_state_dim, self.output_mlp_hidden_dim,
                                     self.output_embedder, self.model, decoder_builder)

    # To use a residual decoder:
    # decoder = MlpSoftmaxDecoder(4, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model,
    #                             lambda layers, input_dim, hidden_dim, model,
    #                               residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.LSTMBuilder))

    self.translator = DefaultTranslator(self.encoder, self.attender, self.decoder)
    self.model_params = ModelParams(self.encoder, self.attender, self.decoder, self.input_reader.vocab.i2w,
                                    self.output_reader.vocab.i2w)

    # single mode
    if args.batch_size is None or args.batch_size == 1 or args.batch_strategy.lower() == 'none':
      print('Start training in non-minibatch mode...')
      self.logger = NonBatchLogger(args.eval_every, total_train_sent)

    # minibatch mode
    else:
      print('Start training in minibatch mode...')
      self.batcher = Batcher.select_batcher(args.batch_strategy)(args.batch_size)
      if args.input_format == "contvec":
        self.batcher.pad_token = np.zeros(self.input_word_emb_dim)
      self.train_corpus_source, self.train_corpus_target = self.batcher.pack(self.train_corpus_source,
                                                                             self.train_corpus_target)
      self.dev_corpus_source, self.dev_corpus_target = self.batcher.pack(self.dev_corpus_source,
                                                                         self.dev_corpus_target)
      self.logger = BatchLogger(args.eval_every, total_train_sent)

  def run_epoch(self):
    self.logger.new_epoch()

    for batch_num, (src, tgt) in enumerate(zip(self.train_corpus_source, self.train_corpus_target)):

      # Loss calculation
      dy.renew_cg()
      loss = self.translator.calc_loss(src, tgt)
      self.logger.update_epoch_loss(src, tgt, loss.value())

      loss.backward()
      self.trainer.update()

      # Devel reporting
      if self.logger.report_train_process():

        self.logger.new_dev()
        for src, tgt in zip(self.dev_corpus_source, self.dev_corpus_target):
          dy.renew_cg()
          loss = self.translator.calc_loss(src, tgt).value()
          self.logger.update_dev_loss(tgt, loss)

        # Write out the model if it's the best one
        if self.logger.report_dev_and_check_model(self.args.model_file):
          self.model_serializer.save_to_file(self.args.model_file, self.model_params, self.model)

        self.trainer.update_epoch()

    return math.exp(self.logger.epoch_loss / self.logger.epoch_words), \
           math.exp(self.logger.dev_loss / self.logger.dev_words)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dynet_mem', type=int)
  parser.add_argument('--dynet_seed', type=int)
  parser.add_argument('--eval_every', type=int)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--batch_strategy', type=str, default='src')
  parser.add_argument('train_source')
  parser.add_argument('train_target')
  parser.add_argument('dev_source')
  parser.add_argument('dev_target')
  parser.add_argument('model_file')
  args = parser.parse_args()
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while True:
    xnmt_trainer.run_epoch()
