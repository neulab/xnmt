# coding: utf-8
import argparse
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
  # TODO: Implement argparse and training
  parser = argparse.ArgumentParser()
  parser.add_argument('train_source')
  parser.add_argument('train_target')
  args = parser.parse_args()

  model = dy.Model()
  trainer = dy.SimpleSGDTrainer(model)

  input_reader = PlainTextReader()
  train_corpus_source = input_reader.read_file(args.train_source)

  output_reader = PlainTextReader()
  train_corpus_target = output_reader.read_file(args.train_target)

  input_vocab_size = output_vocab_size = 31
  input_word_emb_dim = output_word_emb_dim = output_state_dim = attender_hidden_dim = \
  output_mlp_hidden_dim = 17
  encoder_hidden_dim = 26

  input_embedder = SimpleWordEmbedder(len(input_reader.vocab), input_word_emb_dim, model)
  output_embedder = SimpleWordEmbedder(len(output_reader.vocab), output_word_emb_dim, model)
  encoder = BiLstmEncoder(2, encoder_hidden_dim, input_embedder, model)
  attender = StandardAttender(encoder_hidden_dim, output_state_dim, attender_hidden_dim, model)
  decoder = MlpSoftmaxDecoder(2, encoder_hidden_dim, output_state_dim, output_mlp_hidden_dim, output_embedder, model)

  translator = DefaultTranslator(encoder, attender, decoder)
  while True:
    epoch_loss = 0.0
    for src, tgt in zip(train_corpus_source, train_corpus_target):
      dy.renew_cg()
      loss = translator.calc_loss(src, tgt)
      epoch_loss += loss.value()
      loss.backward()
      trainer.update()
    print epoch_loss
