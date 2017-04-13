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
from options import Option, OptionParser, general_options

'''
This will be the main class to perform training.
'''

options = [
  Option("eval_every", int, default_value=1000, force_flag=True),
  Option("batch_size", int, default_value=32, force_flag=True),
  Option("batch_strategy", default_value="src"),
  Option("train_source"),
  Option("train_target"),
  Option("dev_source"),
  Option("dev_target"),
  Option("model_file"),
  Option("input_type", default_value="word"),
  Option("input_word_embed_dim", int, default_value=67),
  Option("output_word_embed_dim", int, default_value=67),
  Option("output_state_dim", int, default_value=67),
  Option("attender_hidden_dim", int, default_value=67),
  Option("output_mlp_hidden_dim", int, default_value=67),
  Option("encoder_hidden_dim", int, default_value=64),
  Option("trainer", default_value="sgd"),
  Option("eval_metrics", default_value="bleu"),
  Option("encoder_layers", int, default_value=2),
  Option("decoder_layers", int, default_value=2),
  Option("encoder_type", default_value="BiLSTM"),
  Option("decoder_type", default_value="LSTM"),
]

class XnmtTrainer:
  def __init__(self, args):
    dy.renew_cg()

    self.args = args  # save for later

    self.model = dy.Model()

    if args.trainer.lower() == "sgd":
      self.trainer = dy.SimpleSGDTrainer(self.model)
    elif args.trainer.lower() == "adam":
      self.trainer = dy.AdamTrainer(self.model)
    else:
      raise RuntimeError("Unkonwn trainer {}".format(args.trainer))

    encoder_type = args.encoder_type.lower()
    if encoder_type == "BiLSTM".lower():
      encoder_builder = BiLSTMEncoder
    elif encoder_type == "ResidualLSTM".lower():
      encoder_builder = ResidualLSTMEncoder
    elif encoder_type == "ResidualBiLSTM".lower():
      encoder_builder = ResidualBiLSTMEncoder
    elif encoder_type == "PyramidalBiLSTM".lower():
      encoder_builder = PyramidalBiLSTMEncoder
    else:
      raise RuntimeError("Unkonwn encoder type {}".format(encoder_type))

    decoder_type = args.decoder_type.lower()
    if decoder_type == "LSTM".lower():
      decoder_builder = dy.LSTMBuilder
    elif decoder_type == "ResidualLSTM".lower():
      decoder_builder = lambda num_layers, input_dim, hidden_dim, model: \
        residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim, model, dy.LSTMBuilder)
    else:
      raise RuntimeError("Unkonwn decoder type {}".format(encoder_type))

    # Create the model serializer
    self.model_serializer = JSONSerializer()

    # Read in training and dev corpora
    self.input_reader = InputReader.create_input_reader(args.input_type)
    self.output_reader = InputReader.create_input_reader("word")

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
    self.output_word_emb_dim = args.output_word_embed_dim
    self.output_state_dim = args.output_state_dim
    self.attender_hidden_dim = args.attender_hidden_dim
    self.output_mlp_hidden_dim = args.output_mlp_hidden_dim
    self.encoder_hidden_dim = args.encoder_hidden_dim

    if args.input_type == "word":
      self.input_embedder = SimpleWordEmbedder(len(self.input_reader.vocab), self.input_word_emb_dim, self.model)
    elif args.input_type == "feat-vec":
      self.input_embedder = FeatVecNoopEmbedder(self.input_word_emb_dim, self.model)
    else:
      raise RuntimeError("Unkonwn input type {}".format(args.input_type))
    self.output_embedder = SimpleWordEmbedder(len(self.output_reader.vocab), self.output_word_emb_dim, self.model)
    self.encoder = encoder_builder(self.args.encoder_layers, self.encoder_hidden_dim, self.input_embedder, self.model)
    self.attender = StandardAttender(self.encoder_hidden_dim, self.output_state_dim, self.attender_hidden_dim,
                                     self.model)
    self.decoder = MlpSoftmaxDecoder(self.args.decoder_layers, self.encoder_hidden_dim, self.output_state_dim,
                                     self.output_mlp_hidden_dim,
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
      if args.input_type == "feat-vec":
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
  parser = OptionParser()
  parser.add_task("train", general_options + options)
  args = parser.args_from_command_line("train", sys.argv[1:])
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while True:
    xnmt_trainer.run_epoch()
