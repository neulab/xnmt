import unittest

import _dynet as dy ; dyparams = dy.DynetParams() ; dyparams.set_random_seed(13); dyparams.init()

import xnmt.model_globals as model_globals
from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.encoder import LSTMEncoder, ResidualLSTMEncoder
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader

class TestEncoder(unittest.TestCase):
  
  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(), 
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def assert_in_out_len_equal(self, model):
    dy.renew_cg()
    embeddings = model.src_embedder.embed_sent(self.training_corpus.train_src_data[0])
    encodings = model.encoder.transduce(embeddings)
    self.assertEqual(len(embeddings), len(encodings))
    
  def test_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_res_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=ResidualLSTMEncoder(layers=3),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

if __name__ == '__main__':
  unittest.main()
