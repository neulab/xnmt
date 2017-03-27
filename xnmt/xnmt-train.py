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
  parser.add_argument('--batch_size', dest='minibatch_size', type=int)
  parser.add_argument('--eval_every', dest='eval_every', type=int)
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
    count_tgt_words = lambda tgt_words: len(tgt_words)
    count_sent_num = lambda x: 1

  # minibatch mode
  else:
    print('Start training in minibatch mode...')
    batcher = SourceBucketBatcher(args.minibatch_size)
    train_corpus_source, train_corpus_target = batcher.pack(train_corpus_source, train_corpus_target)
    dev_corpus_source, dev_corpus_target = batcher.pack(dev_corpus_source, dev_corpus_target)
    count_tgt_words = lambda tgt_words: sum(len(x) for x in tgt_words)
    count_sent_num = lambda x: len(x)

  # Main training loop
  epoch_num = 0
  while True:
    epoch_loss = 0.0
    epoch_words = 0
    epoch_num += 1

    sent_num = 0
    sent_num_not_report = 0
    for batch_num, (src, tgt) in enumerate(zip(train_corpus_source, train_corpus_target)):
      # Loss calculation
      dy.renew_cg()
      batch_sent_num = count_sent_num(src)
      sent_num += batch_sent_num
      sent_num_not_report += batch_sent_num
      loss = translator.calc_loss(src, tgt)
      epoch_words += count_tgt_words(tgt)
      epoch_loss += loss.value()
      loss.backward()
      trainer.update()
      
      print_report = (sent_num_not_report >= args.eval_every) or (sent_num == total_train_sent)
      if print_report:
        while sent_num_not_report >= args.eval_every:
          sent_num_not_report -= args.eval_every

      # Training reporting
      fractional_epoch = (epoch_num - 1) + sent_num / total_train_sent

      if print_report:
        print ('Epoch %.4f: train_ppl=%.4f (loss/word=%.4f, words=%d)' % (
          fractional_epoch, math.exp(epoch_loss/epoch_words), epoch_loss/epoch_words, epoch_words))

      # Devel reporting
      if print_report:
        dev_loss = 0.0
        dev_words = 0
        for src, tgt in zip(dev_corpus_source, dev_corpus_target):
          dy.renew_cg()
          loss = translator.calc_loss(src, tgt).value()
          dev_loss += loss
          dev_words += count_tgt_words(tgt)
        print ('Epoch %.4f: devel_ppl=%.4f (loss/word=%.4f, words=%d)' % (
          fractional_epoch, math.exp(dev_loss/dev_words), dev_loss/dev_words, dev_words))

    trainer.update_epoch()
