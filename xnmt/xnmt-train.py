# coding: utf-8
import argparse
import math
import dynet as dy
from embedder import *
from attender import *
from input import *
from encoder import *
from decoder import *
from translator import *
from logger import *
'''
This will be the main class to perform training.
'''

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dynet_mem', type=int)
  parser.add_argument('--batch_size', dest='minibatch_size', type=int)
  parser.add_argument('--eval_every', dest='eval_every', type=int)
  parser.add_argument('--batch_strategy', dest='batch_strategy', type=str)
  parser.add_argument('train_source')
  parser.add_argument('train_target')
  parser.add_argument('dev_source')
  parser.add_argument('dev_target')
  args = parser.parse_args()
  print("Starting xnmt-train:\nArguments: %r" % (args))


  model = dy.Model()
  trainer = dy.SimpleSGDTrainer(model)

  # Read in training and dev corpora
  input_reader = PlainTextReader()
  output_reader = PlainTextReader()

  train_corpus_source = input_reader.read_file(args.train_source)
  train_corpus_target = output_reader.read_file(args.train_target)
  assert len(train_corpus_source) == len(train_corpus_target)
  total_train_sent = len(train_corpus_source)

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
  encoder = BiLSTMEncoder(2, encoder_hidden_dim, input_embedder, model)
  attender = StandardAttender(encoder_hidden_dim, output_state_dim, attender_hidden_dim, model)
  decoder = MlpSoftmaxDecoder(2, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model)

  translator = DefaultTranslator(encoder, attender, decoder)


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

  while True:

    logger.new_epoch()

    for batch_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):

      # Loss calculation
      dy.renew_cg()
      loss = translator.calc_loss(src, tgt)
      logger.update_epoch_loss(src, tgt, loss.value())

      loss.backward()
      trainer.update()
      
      dev_report = logger.print_train_report()

      # Devel reporting
      if dev_report:
        logger.new_dev()
        for src, tgt in zip(dev_corpus_source, dev_corpus_target):
          dy.renew_cg()
          loss = translator.calc_loss(src, tgt).value()
          logger.update_dev_loss(tgt, loss)
        logger.print_dev_report()

    trainer.update_epoch()
