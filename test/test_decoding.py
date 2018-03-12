import unittest

import dynet_config
import dynet as dy

from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
from xnmt.input_reader import PlainTextReader
from xnmt.loss_calculator import LossCalculator
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.translator import DefaultTranslator
from xnmt.exp_global import ExpGlobal, PersistentParamCollection

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    xnmt.events.clear()
    self.exp_global = ExpGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(exp_global=self.exp_global),
              attender=MlpAttender(exp_global=self.exp_global),
              trg_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(exp_global=self.exp_global, vocab_size=100, bridge=CopyBridge(exp_global=self.exp_global, dec_layers=1)),
            )
    self.model.set_train(False)
    self.model.initialize_generator()

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def assert_forced_decoding(self, sent_id):
    dy.renew_cg()
    outputs = self.model.generate_output(self.src_data[sent_id], sent_id,
                                         forced_trg_ids=self.trg_data[sent_id])
    self.assertItemsEqual(self.trg_data[sent_id], outputs[0].actions)

  def test_forced_decoding(self):
    for i in range(1):
      self.assert_forced_decoding(sent_id=i)

class TestForcedDecodingLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.exp_global = ExpGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(exp_global=self.exp_global),
              attender=MlpAttender(exp_global=self.exp_global),
              trg_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(exp_global=self.exp_global, vocab_size=100, bridge=CopyBridge(exp_global=self.exp_global, dec_layers=1)),
            )
    self.model.set_train(False)
    self.model.initialize_generator()

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=self.trg_data[0],
                                      loss_calculator=LossCalculator()).value()
    dy.renew_cg()
    self.model.initialize_generator()
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score = outputs[0].score
    self.assertAlmostEqual(-output_score, train_loss, places=5)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.exp_global = ExpGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(exp_global=self.exp_global),
              attender=MlpAttender(exp_global=self.exp_global),
              trg_embedder=SimpleWordEmbedder(exp_global=self.exp_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(exp_global=self.exp_global, vocab_size=100, bridge=CopyBridge(exp_global=self.exp_global, dec_layers=1)),
            )
    self.model.set_train(False)
    self.model.initialize_generator()

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    self.model.initialize_generator()
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score = outputs[0].score

    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=outputs[0].actions,
                                      loss_calculator=LossCalculator()).value()

    self.assertAlmostEqual(-output_score, train_loss, places=5)


if __name__ == '__main__':
  unittest.main()
