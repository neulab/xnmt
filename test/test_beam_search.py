import os
import unittest

import dynet as dy

from xnmt.modelparts.attenders import MlpAttender
from xnmt import batchers, event_trigger, events
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.modelparts.transforms import NonLinear
from xnmt.modelparts.scorers import Softmax
from xnmt.models.translators import DefaultTranslator
from xnmt.search_strategies import BeamSearch, GreedySearch
from xnmt.param_collections import ParamManager
from xnmt.persistence import LoadSerialized, initialize_if_needed, YamlPreloader
from xnmt.vocabs import Vocab

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    events.clear()
    ParamManager.init_param_col()

    # Load a pre-trained model
    load_experiment = LoadSerialized(
      filename=f"examples/data/tiny_jaen.model",
      overwrite=[
        {"path" : "train", "val" : None},
        {"path": "status", "val": None},
      ])
    EXP_DIR = '.'
    EXP = "decode"
    uninitialized_experiment = YamlPreloader.preload_obj(load_experiment, exp_dir=EXP_DIR, exp_name=EXP)
    loaded_experiment = initialize_if_needed(uninitialized_experiment)
    ParamManager.populate()

    # Pull out the parts we need from the experiment
    self.model = loaded_experiment.model
    src_vocab = self.model.src_reader.vocab
    trg_vocab = self.model.trg_reader.vocab

    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), BeamSearch())

    # Make sure the output of beam search is the same as the target sentence
    # (this is a very overfit model on exactly this data)
    self.assertEqual(outputs[0].sent_len(), self.trg_data[0].sent_len())
    for i in range(outputs[0].sent_len()):
      self.assertEqual(outputs[0][i], self.trg_data[0][i])

    # Verify that the loss we get from beam search is the same as the loss
    # we get if we call model.calc_nll
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
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))

  def test_greedy_vs_beam(self):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), BeamSearch(beam_size=1))
    output_score1 = outputs[0].score

    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())
    output_score2 = outputs[0].score

    self.assertAlmostEqual(output_score1, output_score2)


if __name__ == '__main__':
  unittest.main()
