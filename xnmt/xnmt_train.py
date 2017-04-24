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
from loss_tracker import *
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
  Option("pretrained_model_file", default_value="", help="path of pre-trained model file"),
  Option("input_format", default_value="text", help="format of input data: text/contvec"),
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
  Option("residual_to_output", bool, default_value=True,
         help="If using residual networks, whether to add a residual connection to the output layer"),
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
      self.encoder_builder = BiLSTMEncoder
    elif encoder_type == "ResidualLSTM".lower():
      self.encoder_builder = lambda encoder_layers, encoder_hidden_dim, input_embedder, model:\
        ResidualLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model, args.residual_to_output)
    elif encoder_type == "ResidualBiLSTM".lower():
      self.encoder_builder = lambda encoder_layers, encoder_hidden_dim, input_embedder, model:\
        ResidualBiLSTMEncoder(encoder_layers, encoder_hidden_dim, input_embedder, model, args.residual_to_output)
    elif encoder_type == "PyramidalBiLSTM".lower():
      self.encoder_builder = PyramidalBiLSTMEncoder
    elif encoder_type == "ConvBiLSTM".lower():
      self.encoder_builder = ConvBiLSTMEncoder
    else:
      raise RuntimeError("Unkonwn encoder type {}".format(encoder_type))

    decoder_type = args.decoder_type.lower()
    if decoder_type == "LSTM".lower():
      self.decoder_builder = dy.VanillaLSTMBuilder
    elif decoder_type == "ResidualLSTM".lower():
      self.decoder_builder = lambda num_layers, input_dim, hidden_dim, model: \
        residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, args.residual_to_output)

    else:
      raise RuntimeError("Unkonwn decoder type {}".format(encoder_type))

    # Create the model serializer
    self.create_model()
    # single mode
    if args.batch_size is None or args.batch_size == 1 or args.batch_strategy.lower() == 'none':
      print('Start training in non-minibatch mode...')
      self.logger = NonBatchLossTracker(args.eval_every, self.total_train_sent)

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
      self.logger = BatchLossTracker(args.eval_every, self.total_train_sent)

  def create_model(self):
    if self.args.pretrained_model_file:
      self.model_serializer = JSONSerializer()
      self.model_params = self.model_serializer.load_from_file(self.args.pretrained_model_file, self.model)
      source_vocab = Vocab(self.model_params.source_vocab)
      target_vocab = Vocab(self.model_params.target_vocab)
      self.encoder = self.model_params.encoder
      self.attender = self.model_params.attender
      self.decoder = self.model_params.decoder
      self.translator = DefaultTranslator(self.encoder, self.attender, self.decoder)
      self.input_reader = InputReader.create_input_reader(self.args.input_format, source_vocab)
      self.output_reader = InputReader.create_input_reader("text", target_vocab)
      self.read_data()
      return

    self.model_serializer = JSONSerializer()
    # Read in training and dev corpora

    self.input_reader = InputReader.create_input_reader(self.args.input_format)
    self.output_reader = InputReader.create_input_reader("text")
    self.read_data()
    # Create the translator object and all its subparts
    self.input_word_emb_dim = self.args.input_word_embed_dim
    self.output_word_emb_dim = self.args.output_word_embed_dim
    self.output_state_dim = self.args.output_state_dim
    self.attender_hidden_dim = self.args.attender_hidden_dim
    self.output_mlp_hidden_dim = self.args.output_mlp_hidden_dim
    self.encoder_hidden_dim = self.args.encoder_hidden_dim

    if self.args.input_format == "text":
      self.input_embedder = SimpleWordEmbedder(len(self.input_reader.vocab), self.input_word_emb_dim, self.model)
    elif self.args.input_format == "contvec":
      self.input_embedder = FeatVecNoopEmbedder(self.input_word_emb_dim, self.model)
    else:
      raise RuntimeError("Unkonwn input type {}".format(self.args.input_format))


    self.output_embedder = SimpleWordEmbedder(len(self.output_reader.vocab), self.output_word_emb_dim, self.model)
    self.encoder = self.encoder_builder(self.args.encoder_layers, self.encoder_hidden_dim, self.input_embedder, self.model)
    self.attender = StandardAttender(self.encoder_hidden_dim, self.output_state_dim, self.attender_hidden_dim,
                                     self.model)
    self.decoder = MlpSoftmaxDecoder(self.args.decoder_layers, self.encoder_hidden_dim, self.output_state_dim,
                                     self.output_mlp_hidden_dim,
                                     self.output_embedder, self.model, self.decoder_builder)

    # To use a residual decoder:
    # decoder = MlpSoftmaxDecoder(4, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model,
    #                             lambda layers, input_dim, hidden_dim, model,
    #                               residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder))

    self.translator = DefaultTranslator(self.encoder, self.attender, self.decoder)
    self.model_params = ModelParams(self.encoder, self.attender, self.decoder, self.input_reader.vocab.i2w,
                                    self.output_reader.vocab.i2w)



  def read_data(self):
    self.train_corpus_source = self.input_reader.read_file(self.args.train_source)
    self.train_corpus_target = self.output_reader.read_file(self.args.train_target)
    assert len(self.train_corpus_source) == len(self.train_corpus_target)
    self.total_train_sent = len(self.train_corpus_source)
    if self.args.eval_every == None:
      self.args.eval_every = self.total_train_sent

    self.input_reader.freeze()
    self.output_reader.freeze()

    self.dev_corpus_source = self.input_reader.read_file(self.args.dev_source)
    self.dev_corpus_target = self.output_reader.read_file(self.args.dev_target)
    assert len(self.dev_corpus_source) == len(self.dev_corpus_target)


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
