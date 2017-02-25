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
'''
This will be the main class to perform training.
'''

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dynet_mem', type=int)
  parser.add_argument('train_source')
  parser.add_argument('train_target')
  parser.add_argument('dev_source')
  parser.add_argument('dev_target')
  args = parser.parse_args()

  model = dy.Model()
  trainer = dy.SimpleSGDTrainer(model)

  # Read in training and dev corpora
  input_reader = PlainTextReader()
  output_reader = PlainTextReader()

  train_corpus_source = input_reader.read_file(args.train_source)
  train_corpus_target = output_reader.read_file(args.train_target)
  assert len(train_corpus_source) == len(train_corpus_target)

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
  encoder = BiLstmEncoder(2, encoder_hidden_dim, input_embedder, model)
  attender = StandardAttender(encoder_hidden_dim, output_state_dim, attender_hidden_dim, model)
  decoder = MlpSoftmaxDecoder(2, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model)

  translator = DefaultTranslator(encoder, attender, decoder)

  # Main training loop
  epoch_num = 0
  while True:
    epoch_loss = 0.0
    word_count = 0
    epoch_num += 1
    for sent_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):
      dy.renew_cg()
      loss = translator.calc_loss(src, tgt)
      word_count += len(tgt)
      epoch_loss += loss.value()
      loss.backward()
      trainer.update()

      if sent_num % 100 == 99 or sent_num == len(train_corpus_source) - 1:
        dev_loss = 0.0
        dev_words = 0
        for src, tgt in zip(dev_corpus_source, dev_corpus_target):
          dy.renew_cg()
          loss = translator.calc_loss(src, tgt).value()
          dev_loss += loss
          dev_words += len(tgt)
        print ((epoch_num - 1) + 1.0 * (sent_num + 1) / len(train_corpus_source),
               'Dev perplexity:', math.exp(dev_loss / dev_words),
               '(%f over %d words)' % (dev_loss, dev_words))
    trainer.update_epoch()
    print (epoch_num, 'Train perplexity:', math.exp(epoch_loss/word_count), '(%f over %d words)' % (epoch_loss, word_count))
