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
  Option("pretrained_model_file", default_value="", help="Path of pre-trained model file"),
  Option("input_vocab", default_value="", help="Path of fixed input vocab file"),
  Option("output_vocab", default_value="", help="Path of fixed output vocab file"),
  Option("input_format", default_value="text", help="Format of input data: text/contvec"),
  Option("default_layer_dim", int, default_value=512, help="Default size to use for layers if not otherwise overridden"),
  Option("input_word_embed_dim", int, required=False),
  Option("output_word_embed_dim", int, required=False),
  Option("output_state_dim", int, required=False),
  Option("output_mlp_hidden_dim", int, required=False),
  Option("attender_hidden_dim", int, required=False),
  Option("encoder_hidden_dim", int, required=False),
  Option("trainer", default_value="sgd"),
  Option("learning_rate", float, default_value=0.1),
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
      self.trainer = dy.SimpleSGDTrainer(self.model, e0 = args.learning_rate)
    elif args.trainer.lower() == "adam":
      self.trainer = dy.AdamTrainer(self.model, alpha = args.learning_rate)
    else:
      raise RuntimeError("Unkonwn trainer {}".format(args.trainer))

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
        self.batcher.pad_token = np.zeros(self.encoder.embedder.get_embed_dim())
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
      self.input_reader.freeze()
      self.output_reader.freeze()
      self.read_data()
      return

    self.model_serializer = JSONSerializer()

    # Read in training and dev corpora
    input_vocab, output_vocab = None, None
    if self.args.input_vocab:
      input_vocab = Vocab(vocab_file=self.args.input_vocab)
    if self.args.output_vocab:
      output_vocab = Vocab(vocab_file=self.args.output_vocab)
    self.input_reader = InputReader.create_input_reader(self.args.input_format, input_vocab)
    self.output_reader = InputReader.create_input_reader("text", output_vocab)
    if self.args.input_vocab:
      self.input_reader.freeze()
    if self.args.output_vocab:
      self.output_reader.freeze()
    self.read_data()
    
    # Get layer sizes: replace by default if not specified
    for opt in ["input_word_embed_dim", "output_word_embed_dim", "output_state_dim", "output_mlp_hidden_dim",
                "encoder_hidden_dim", "attender_hidden_dim"]:
      if getattr(self.args, opt) is None:
        setattr(self.args, opt, self.args.default_layer_dim)

    self.input_word_emb_dim = self.args.input_word_embed_dim
    self.output_word_emb_dim = self.args.output_word_embed_dim
    self.output_state_dim = self.args.output_state_dim
    self.attender_hidden_dim = self.args.attender_hidden_dim
    self.output_mlp_hidden_dim = self.args.output_mlp_hidden_dim
    self.encoder_hidden_dim = self.args.encoder_hidden_dim

    self.input_embedder = Embedder.from_spec(self.args.input_format, len(self.input_reader.vocab),
                                             self.input_word_emb_dim, self.model)

    self.output_embedder = SimpleWordEmbedder(len(self.output_reader.vocab), self.output_word_emb_dim, self.model)

    self.encoder = Encoder.from_spec(self.args.encoder_type, self.args.encoder_layers, self.encoder_hidden_dim,
                                     self.input_embedder, self.model, self.args.residual_to_output)

    self.attender = StandardAttender(self.encoder_hidden_dim, self.output_state_dim, self.attender_hidden_dim,
                                     self.model)

    decoder_rnn = Decoder.rnn_from_spec(self.args.decoder_type, self.args.decoder_layers, self.encoder_hidden_dim,
                                        self.output_state_dim, self.model, self.args.residual_to_output)
    self.decoder = MlpSoftmaxDecoder(self.args.decoder_layers, self.encoder_hidden_dim, self.output_state_dim,
                                     self.output_mlp_hidden_dim,
                                     self.output_embedder, self.model, decoder_rnn)

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
