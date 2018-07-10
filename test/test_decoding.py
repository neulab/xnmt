import unittest

import dynet_config
import dynet as dy

from xnmt.attender import MlpAttender
import xnmt.batcher
from xnmt.bridge import CopyBridge
from xnmt.decoder import AutoRegressiveDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
from xnmt.input_reader import PlainTextReader
from xnmt.loss_calculator import AutoRegressiveMLELoss
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.param_collection import ParamManager
from xnmt.transform import NonLinear
from xnmt.translator import DefaultTranslator
from xnmt.scorer import Softmax
from xnmt.search_strategy import GreedySearch

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

    self.search = GreedySearch()

  def assert_forced_decoding(self, sent_id):
    dy.renew_cg()
    outputs = self.model.generate(xnmt.batcher.mark_as_batch([self.src_data[sent_id]]), [sent_id], self.search,
                                  forced_trg_ids=xnmt.batcher.mark_as_batch([self.trg_data[sent_id]]))
    self.assertItemsEqual(self.trg_data[sent_id].words, outputs[0].actions)

  def test_forced_decoding(self):
    for i in range(1):
      self.assert_forced_decoding(sent_id=i)

class TestForcedDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=self.trg_data[0],
                                      loss_calculator=AutoRegressiveMLELoss()).value()
    dy.renew_cg()
    outputs = self.model.generate(xnmt.batcher.mark_as_batch([self.src_data[0]]), [0], GreedySearch(),
                                  forced_trg_ids=xnmt.batcher.mark_as_batch([self.trg_data[0]]))
    output_score = outputs[0].score
    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    outputs = self.model.generate(xnmt.batcher.mark_as_batch([self.src_data[0]]), [0], GreedySearch(),
                                  forced_trg_ids=xnmt.batcher.mark_as_batch([self.trg_data[0]]))
    output_score = outputs[0].score

    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=outputs[0],
                                      loss_calculator=AutoRegressiveMLELoss()).value()

    self.assertAlmostEqual(-output_score, train_loss, places=5)


if __name__ == '__main__':
  unittest.main()
