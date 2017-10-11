import unittest

import dynet as dy
import numpy as np

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.encoder import LSTMEncoder, PyramidalLSTMEncoder
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder, CopyBridge
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader
from xnmt.batcher import mark_as_batch, Mask
import xnmt.xnmt_train
from xnmt.options import Args
from xnmt.vocab import Vocab
from xnmt.model_context import ModelContext, PersistentParamCollection

class TestTruncatedBatchTraining(unittest.TestCase):

  def setUp(self):
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def assert_single_loss_equals_batch_loss(self, model, batch_size=5):
    """
    Tests whether single loss equals batch loss.
    Truncating src / trg sents to same length so no masking is necessary
    """
    batch_size = 5
    src_sents = self.training_corpus.train_src_data[:batch_size]
    src_min = min([len(x) for x in src_sents])
    src_sents_trunc = [s[:src_min] for s in src_sents]
    for single_sent in src_sents_trunc: single_sent[src_min-1] = Vocab.ES
    trg_sents = self.training_corpus.train_trg_data[:batch_size]
    trg_min = min([len(x) for x in trg_sents])
    trg_sents_trunc = [s[:trg_min] for s in trg_sents]
    for single_sent in trg_sents_trunc: single_sent[trg_min-1] = Vocab.ES

    single_loss = 0.0
    for sent_id in range(batch_size):
      dy.renew_cg()
      train_loss = model.calc_loss(src=src_sents_trunc[sent_id],
                                        trg=trg_sents_trunc[sent_id]).value()
      single_loss += train_loss

    dy.renew_cg()

    batched_loss = model.calc_loss(src=mark_as_batch(src_sents_trunc),
                                        trg=mark_as_batch(trg_sents_trunc)).value()
    self.assertAlmostEqual(single_loss, sum(batched_loss), places=4)

  def test_loss_model1(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMEncoder(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=PyramidalLSTMEncoder(self.model_context, layers=3),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model3(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMEncoder(self.model_context, layers=3),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100, bridge=CopyBridge(self.model_context, dec_layers=1)),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

class TestBatchTraining(unittest.TestCase):

  def setUp(self):
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def assert_single_loss_equals_batch_loss(self, model, batch_size=5):
    """
    Tests whether single loss equals batch loss.
    Here we don't truncate the target side and use masking.
    """
    batch_size = 5
    src_sents = self.training_corpus.train_src_data[:batch_size]
    src_min = min([len(x) for x in src_sents])
    src_sents_trunc = [s[:src_min] for s in src_sents]
    for single_sent in src_sents_trunc: single_sent[src_min-1] = Vocab.ES
    trg_sents = self.training_corpus.train_trg_data[:batch_size]
    trg_max = max([len(x) for x in trg_sents])
    trg_masks = Mask(np.zeros([batch_size, trg_max]))
    for i in range(batch_size):
      for j in range(len(trg_sents[i]), trg_max):
        trg_masks.np_arr[i,j] = 1.0
    trg_sents_padded = [[w for w in s] + [Vocab.ES]*(trg_max-len(s)) for s in trg_sents]

    single_loss = 0.0
    for sent_id in range(batch_size):
      dy.renew_cg()
      train_loss = model.calc_loss(src=src_sents_trunc[sent_id],
                                   trg=trg_sents[sent_id]).value()
      single_loss += train_loss

    dy.renew_cg()

    batched_loss = model.calc_loss(src=mark_as_batch(src_sents_trunc),
                                   trg=mark_as_batch(trg_sents_padded, trg_masks)).value()
    self.assertAlmostEqual(single_loss, sum(batched_loss), places=4)

  def test_loss_model1(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMEncoder(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=PyramidalLSTMEncoder(self.model_context, layers=3),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model3(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMEncoder(self.model_context, layers=3),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100, bridge=CopyBridge(self.model_context, dec_layers=1)),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)


class TestTrainDevLoss(unittest.TestCase):

  def test_train_dev_loss_equal(self):
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    task_options = xnmt.xnmt_train.options
    train_args = dict({opt.name: opt.default_value for opt in task_options if
                                opt.default_value is not None or not opt.required})
    train_args['training_corpus'] = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                                            train_trg = "examples/data/head.en",
                                                            dev_src = "examples/data/head.ja",
                                                            dev_trg = "examples/data/head.en")
    train_args['corpus_parser'] = BilingualCorpusParser(src_reader = PlainTextReader(),
                                                        trg_reader = PlainTextReader())
    train_args['model'] = DefaultTranslator(src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
                                            encoder=LSTMEncoder(self.model_context),
                                            attender=StandardAttender(self.model_context),
                                            trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
                                            decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
                                            )
    train_args['model_file'] = None
    train_args['save_num_checkpoints'] = 0
    xnmt_trainer = xnmt.xnmt_train.XnmtTrainer(args=Args(**train_args), need_deserialization=False, param_collection=self.model_context.dynet_param_collection)
    xnmt_trainer.model_context = self.model_context
    xnmt_trainer.run_epoch(update_weights=False)
    self.assertAlmostEqual(xnmt_trainer.logger.epoch_loss.loss_values['loss'] / xnmt_trainer.logger.epoch_words,
                           xnmt_trainer.logger.dev_score.loss)

class TestOverfitting(unittest.TestCase):

  def test_overfitting(self):
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.model_context.default_layer_dim = 16
    task_options = xnmt.xnmt_train.options
    train_args = dict({opt.name: opt.default_value for opt in task_options if
                                opt.default_value is not None or not opt.required})
    train_args['training_corpus'] = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                                            train_trg = "examples/data/head.en",
                                                            dev_src = "examples/data/head.ja",
                                                            dev_trg = "examples/data/head.en")
    train_args['corpus_parser'] = BilingualCorpusParser(src_reader = PlainTextReader(),
                                                        trg_reader = PlainTextReader())
    train_args['model'] = DefaultTranslator(src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
                                            encoder=LSTMEncoder(self.model_context),
                                            attender=StandardAttender(self.model_context),
                                            trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
                                            decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
                                            )
    train_args['model_file'] = None
    train_args['save_num_checkpoints'] = 0
    train_args['trainer'] = "adam"
    train_args['learning_rate'] = 0.1
    xnmt_trainer = xnmt.xnmt_train.XnmtTrainer(args=Args(**train_args), need_deserialization=False, param_collection=self.model_context.dynet_param_collection)
    xnmt_trainer.model_context = self.model_context
    for _ in range(50):
      xnmt_trainer.run_epoch(update_weights=True)
    self.assertAlmostEqual(0.0,
                           xnmt_trainer.logger.epoch_loss.loss_values['loss'] / xnmt_trainer.logger.epoch_words,
                           places=2)

if __name__ == '__main__':
  unittest.main()
