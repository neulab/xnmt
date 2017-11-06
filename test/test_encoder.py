import unittest

import dynet_config
import dynet as dy

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.lstm import LSTMSeqTransducer
from xnmt.residual import ResidualLSTMSeqTransducer
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader
from xnmt.model_context import ModelContext, PersistentParamCollection
import xnmt.events

class TestEncoder(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext()
    self.model_context.dynet_param_collection = PersistentParamCollection("some_file", 1)
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
    encodings = model.encoder(embeddings)
    self.assertEqual(len(embeddings), len(encodings))

  def test_uni_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context, layers=3, bidirectional=True),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_bi_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=LSTMSeqTransducer(self.model_context, layers=1, bidirectional=False),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_res_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=ResidualLSTMSeqTransducer(self.model_context, layers=3),
              attender=StandardAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

if __name__ == '__main__':
  unittest.main()
