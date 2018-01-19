import unittest

import dynet_config
import dynet as dy

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.residual import ResidualLSTMSeqTransducer
from xnmt.attender import MlpAttender
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.input import PlainTextReader
from xnmt.model_context import ModelContext, PersistentParamCollection
import xnmt.events

class TestEncoder(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.model_context = ModelContext(dynet_param_collection=PersistentParamCollection("some_file", 1))

    self.src_reader = PlainTextReader()
    self.trg_reader = PlainTextReader()
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))

  @xnmt.events.register_xnmt_event
  def set_train(self, val):
    pass
  @xnmt.events.register_xnmt_event
  def start_sent(self, src):
    pass

  def assert_in_out_len_equal(self, model):
    dy.renew_cg()
    self.set_train(True)
    src = self.src_data[0]
    self.start_sent(src)
    embeddings = model.src_embedder.embed_sent(src)
    encodings = model.encoder(embeddings)
    self.assertEqual(len(embeddings), len(encodings))

  def test_bi_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_reader=self.src_reader,
              trg_reader=self.trg_reader,
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=BiLSTMSeqTransducer(self.model_context, layers=3),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_uni_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_reader=self.src_reader,
              trg_reader=self.trg_reader,
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=UniLSTMSeqTransducer(self.model_context),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

  def test_res_lstm_encoder_len(self):
    model = DefaultTranslator(
              src_reader=self.src_reader,
              trg_reader=self.trg_reader,
              src_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              encoder=ResidualLSTMSeqTransducer(self.model_context, layers=3),
              attender=MlpAttender(self.model_context),
              trg_embedder=SimpleWordEmbedder(self.model_context, vocab_size=100),
              decoder=MlpSoftmaxDecoder(self.model_context, vocab_size=100),
            )
    self.assert_in_out_len_equal(model)

if __name__ == '__main__':
  unittest.main()
