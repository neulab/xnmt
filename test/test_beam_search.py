import unittest

import dynet_config
import dynet as dy

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.lstm import LSTMSeqTransducer
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder, CopyBridge
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader
from xnmt.model_context import ModelContext, PersistentParamCollection
from xnmt.training_strategy import TrainingStrategy
import xnmt.events

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.model.initialize_training_strategy(TrainingStrategy())
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def assert_forced_decoding(self, sent_id):
    dy.renew_cg()
    outputs = self.model.generate_output(self.training_corpus.train_src_data[sent_id], sent_id,
                                         forced_trg_ids=self.training_corpus.train_trg_data[sent_id])
    self.assertItemsEqual(self.training_corpus.train_trg_data[sent_id], outputs[0][0])

  def test_forced_decoding(self):
    for i in range(1):
      self.assert_forced_decoding(sent_id=i)

class TestForcedDecodingLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100, bridge=CopyBridge(self.model_context, dec_layers=1)),
            )
    self.model.initialize_training_strategy(TrainingStrategy())
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.training_corpus.train_src_data[0],
                                      trg=self.training_corpus.train_trg_data[0]).value()
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.training_corpus.train_src_data[0], 0,
                                         forced_trg_ids=self.training_corpus.train_trg_data[0])
    output_score = outputs[0][1]
    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100, bridge=CopyBridge(self.model_context, dec_layers=1)),
            )
    self.model.initialize_training_strategy(TrainingStrategy())
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def test_single(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.training_corpus.train_src_data[0], 0,
                                         forced_trg_ids=self.training_corpus.train_trg_data[0])
    output_score = outputs[0][1]

    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.training_corpus.train_src_data[0],
                                      trg=outputs[0][0]).value()

    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestGreedyVsBeam(unittest.TestCase):
  """
  Test if greedy search produces same output as beam search with beam 1.
  """
  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100, bridge=CopyBridge(self.model_context, dec_layers=1)),
            )
    self.model.initialize_training_strategy(TrainingStrategy())
    self.model.set_train(False)

    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(),
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def test_greedy_vs_beam(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.training_corpus.train_src_data[0], 0,
                                         forced_trg_ids=self.training_corpus.train_trg_data[0])
    output_score1 = outputs[0][1]

    dy.renew_cg()
    self.model.initialize_generator()
    outputs = self.model.generate_output(self.training_corpus.train_src_data[0], 0,
                                         forced_trg_ids=self.training_corpus.train_trg_data[0])
    output_score2 = outputs[0][1]

    self.assertAlmostEqual(output_score1, output_score2)


if __name__ == '__main__':
  unittest.main()
