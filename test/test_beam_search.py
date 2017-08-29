import unittest

import dynet_config
import dynet as dy

import xnmt.model_globals as model_globals
from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.encoder import LSTMEncoder
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder, CopyBridge
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader

class TestForcedDecodingOutputs(unittest.TestCase):
  
  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
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
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100, bridge=CopyBridge(dec_layers=1)),
            )
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
                                      trg=self.training_corpus.train_trg_data[0],
                                      src_mask=None, trg_mask=None).value()
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.training_corpus.train_src_data[0], 0, 
                                         forced_trg_ids=self.training_corpus.train_trg_data[0])
    output_score = outputs[0][1]
    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100, bridge=CopyBridge(dec_layers=1)),
            )
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
                                      trg=outputs[0][0],
                                      src_mask=None, trg_mask=None).value()

    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestGreedyVsBeam(unittest.TestCase):
  """
  Test if greedy search produces same output as beam search with beam 1.
  """
  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100, bridge=CopyBridge(dec_layers=1)),
            )
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
