import unittest

import dynet as dy

from xnmt.modelparts.attenders import MlpAttender
from xnmt import batchers, event_trigger, events
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculators import MLELoss
from xnmt.modelparts.transforms import NonLinear
from xnmt.modelparts.scorers import Softmax
from xnmt.models.translators import DefaultTranslator
from xnmt.search_strategies import BeamSearch, GreedySearch
from xnmt.param_collections import ParamManager
from xnmt.vocabs import Vocab

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    layer_dim = 512
    events.clear()
    ParamManager.init_param_col()
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(vocab=src_vocab),
      trg_reader=PlainTextReader(vocab=trg_vocab),
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
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def assert_forced_decoding(self, sent_id):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[sent_id]]), [sent_id], BeamSearch(),
                                  forced_trg_ids=batchers.mark_as_batch([self.trg_data[sent_id]]))
    self.assertItemsEqual(self.trg_data[sent_id].words, outputs[0].words)

  def test_forced_decoding(self):
    for i in range(1):
      self.assert_forced_decoding(sent_id=i)

class TestForcedDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    events.clear()
    ParamManager.init_param_col()
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(vocab=src_vocab),
      trg_reader=PlainTextReader(vocab=trg_vocab),
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
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_nll(src=self.src_data[0],
                                     trg=self.trg_data[0]).value()
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), [0], BeamSearch(beam_size=1),
                                  forced_trg_ids=batchers.mark_as_batch([self.trg_data[0]]))
    self.assertAlmostEqual(-outputs[0].score, train_loss, places=4)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    events.clear()
    ParamManager.init_param_col()
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(vocab=src_vocab),
      trg_reader=PlainTextReader(vocab=trg_vocab),
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
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), [0], BeamSearch(),
                                  forced_trg_ids=batchers.mark_as_batch([self.trg_data[0]]))
    dy.renew_cg()
    train_loss = self.model.calc_nll(src=self.src_data[0],
                                     trg=outputs[0]).value()

    self.assertAlmostEqual(-outputs[0].score, train_loss, places=4)

class TestGreedyVsBeam(unittest.TestCase):
  """
  Test if greedy search produces same output as beam search with beam 1.
  """
  def setUp(self):
    layer_dim = 512
    events.clear()
    ParamManager.init_param_col()
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(vocab=src_vocab),
      trg_reader=PlainTextReader(vocab=trg_vocab),
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
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_greedy_vs_beam(self):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), [0], BeamSearch(beam_size=1),
                                  forced_trg_ids=batchers.mark_as_batch([self.trg_data[0]]))
    output_score1 = outputs[0].score

    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), [0], GreedySearch(),
                                  forced_trg_ids=batchers.mark_as_batch([self.trg_data[0]]))
    output_score2 = outputs[0].score

    self.assertAlmostEqual(output_score1, output_score2)


if __name__ == '__main__':
  unittest.main()
