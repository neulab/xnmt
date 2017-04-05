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

def xnmt_train(args, run_for_epochs=None, encoder_builder=BiLSTMEncoder, encoder_layers=2,
               decoder_builder=dy.LSTMBuilder, decoder_layers=2):
  dy.renew_cg()

  model = dy.Model()
  trainer = dy.SimpleSGDTrainer(model)

  # Create the model serializer
  model_serializer = JSONSerializer()

  # Read in training and dev corpora
  input_reader = PlainTextReader()
  output_reader = PlainTextReader()

  train_corpus_source = input_reader.read_file(args.train_source)
  train_corpus_target = output_reader.read_file(args.train_target)
  assert len(train_corpus_source) == len(train_corpus_target)
  total_train_sent = len(train_corpus_source)
  if args.eval_every == None:
    args.eval_every = total_train_sent

  input_reader.freeze()
  output_reader.freeze()

  dev_corpus_source = input_reader.read_file(args.dev_source)
  dev_corpus_target = output_reader.read_file(args.dev_target)
  assert len(dev_corpus_source) == len(dev_corpus_target)

  # Create the translator object and all its subparts
  input_word_emb_dim = output_word_emb_dim = output_state_dim = attender_hidden_dim = \
    output_mlp_hidden_dim = 67
  encoder_hidden_dim = 64

  input_embedder = SimpleWordEmbedder(len(input_reader.vocab), input_word_emb_dim, model)
  output_embedder = SimpleWordEmbedder(len(output_reader.vocab), output_word_emb_dim, model)
  encoder = encoder_builder(encoder_layers, encoder_hidden_dim, input_embedder, model)
  attender = StandardAttender(encoder_hidden_dim, output_state_dim, attender_hidden_dim, model)
  decoder = MlpSoftmaxDecoder(decoder_layers, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim,
                              output_embedder, model, decoder_builder)

  # To use a residual decoder:
  # decoder = MlpSoftmaxDecoder(4, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model,
  #                             lambda layers, input_dim, hidden_dim, model:
  #                               residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.LSTMBuilder))

  translator = DefaultTranslator(encoder, attender, decoder)
  model_params = ModelParams(encoder, attender, decoder, input_reader.vocab.i2w, output_reader.vocab.i2w)

  # single mode
  if args.minibatch_size is None:
    print('Start training in non-minibatch mode...')
    logger = NonBatchLogger(args.eval_every, total_train_sent)

  # minibatch mode
  else:
    print('Start training in minibatch mode...')
    batcher = Batcher.select_batcher(args.batch_strategy)(args.minibatch_size)
    train_corpus_source, train_corpus_target = batcher.pack(train_corpus_source, train_corpus_target)
    dev_corpus_source, dev_corpus_target = batcher.pack(dev_corpus_source, dev_corpus_target)
    logger = BatchLogger(args.eval_every, total_train_sent)

  # Main training loop

  while run_for_epochs is None or logger.epoch_num < run_for_epochs:

    logger.new_epoch()

    for batch_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):

      # Loss calculation
      dy.renew_cg()
      loss = translator.calc_loss(src, tgt)
      logger.update_epoch_loss(src, tgt, loss.value())

      loss.backward()
      trainer.update()

      # Devel reporting
      if logger.report_train_process():

        logger.new_dev()
        for src, tgt in zip(dev_corpus_source, dev_corpus_target):
          dy.renew_cg()
          loss = translator.calc_loss(src, tgt).value()
          logger.update_dev_loss(tgt, loss)

        # Write out the model if it's the best one
        if logger.report_dev_and_check_model(args.model_file):
          model_serializer.save_to_file(args.model_file, model_params, model)

    trainer.update_epoch()

  return math.exp(logger.epoch_loss / logger.epoch_words), math.exp(logger.dev_loss / logger.dev_words)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dynet_mem', type=int)
  parser.add_argument('--dynet_seed', type=int)
  parser.add_argument('--batch_size', dest='minibatch_size', type=int)
  parser.add_argument('--eval_every', dest='eval_every', type=int)
  parser.add_argument('--batch_strategy', dest='batch_strategy', type=str)
  parser.add_argument('train_source')
  parser.add_argument('train_target')
  parser.add_argument('dev_source')
  parser.add_argument('dev_target')
  parser.add_argument('model_file')
  args = parser.parse_args()
  print("Starting xnmt-train:\nArguments: %r" % (args))
  xnmt_train(args)


