import unittest

import dynet_config
import dynet as dy

from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.attender import MlpAttender
from xnmt.decoder import MlpSoftmaxDecoder, CopyBridge
from xnmt.input import PlainTextReader
from xnmt.xnmt_global import XnmtGlobal, PersistentParamCollection
from xnmt.loss_calculator import LossCalculator
import xnmt.events

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    xnmt.events.clear()
    self.xnmt_global = XnmtGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(xnmt_global=self.xnmt_global),
              attender=MlpAttender(xnmt_global=self.xnmt_global),
              trg_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(xnmt_global=self.xnmt_global, vocab_size=100),
            )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

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
    self.xnmt_global = XnmtGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(xnmt_global=self.xnmt_global),
              attender=MlpAttender(xnmt_global=self.xnmt_global),
              trg_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(xnmt_global=self.xnmt_global, vocab_size=100, bridge=CopyBridge(xnmt_global=self.xnmt_global, dec_layers=1)),
            )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=self.trg_data[0],
                                      loss_calculator=LossCalculator()).value()
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    self.assertAlmostEqual(-outputs[0].score, train_loss, places=5)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    self.xnmt_global = XnmtGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(xnmt_global=self.xnmt_global),
              attender=MlpAttender(xnmt_global=self.xnmt_global),
              trg_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(xnmt_global=self.xnmt_global, vocab_size=100, bridge=CopyBridge(xnmt_global=self.xnmt_global, dec_layers=1)),
            )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=outputs[0].actions,
                                      loss_calculator=LossCalculator()).value()

    self.assertAlmostEqual(-outputs[0].score, train_loss, places=5)

class TestGreedyVsBeam(unittest.TestCase):
  """
  Test if greedy search produces same output as beam search with beam 1.
  """
  def setUp(self):
    xnmt.events.clear()
    self.xnmt_global = XnmtGlobal(dynet_param_collection=PersistentParamCollection("some_file", 1))
    self.model = DefaultTranslator(
              src_reader=PlainTextReader(),
              trg_reader=PlainTextReader(),
              src_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              encoder=BiLSTMSeqTransducer(xnmt_global=self.xnmt_global),
              attender=MlpAttender(xnmt_global=self.xnmt_global),
              trg_embedder=SimpleWordEmbedder(xnmt_global=self.xnmt_global, vocab_size=100),
              decoder=MlpSoftmaxDecoder(xnmt_global=self.xnmt_global, vocab_size=100, bridge=CopyBridge(xnmt_global=self.xnmt_global, dec_layers=1)),
            )
    self.model.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_greedy_vs_beam(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score1 = outputs[0].score

    dy.renew_cg()
    self.model.initialize_generator()
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score2 = outputs[0].score

    self.assertAlmostEqual(output_score1, output_score2)


if __name__ == '__main__':
  unittest.main()
