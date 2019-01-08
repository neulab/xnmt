import unittest

# import dynet_config
# dynet_config.set(random_seed=3)

import numpy
import random
import dynet as dy

from xnmt.modelparts.attenders import MlpAttender
from xnmt.modelparts.bridges import NoBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
import xnmt.events
from xnmt.eval import metrics
from xnmt import batchers, event_trigger
from xnmt.param_collections import ParamManager
from xnmt.input_readers import PlainTextReader
from xnmt.input_readers import CharFromWordTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer
from xnmt.simultaneous.translators import SimultaneousTranslator
from xnmt.simultaneous.search_strategies import SimultaneousGreedySearch
from xnmt.loss_calculators import MLELoss
from xnmt.modelparts.transforms import AuxNonLinear, Linear
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab
from xnmt.rl.policy_gradient import PolicyGradient
from xnmt.utils import has_cython


class TestSimultaneousTranslation(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 4
    xnmt.events.clear()
    ParamManager.init_param_col()
    
    self.src_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.vocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.en.vocab"))
    self.loss_calculator = MLELoss()
    
    self.model = SimultaneousTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=100, input_dim=layer_dim),
                                    trg_embed_dim=layer_dim,
                                    bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(True)
    
    self.layer_dim = layer_dim
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
  
  def test_train_nll(self):
    mle_loss = MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])

  def test_simult_greedy(self):
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), SimultaneousGreedySearch())
    
  def test_policy(self):
    self.model.policy_learning = PolicyGradient(input_dim=3*self.layer_dim)
    mle_loss = MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])

if __name__ == "__main__":
  unittest.main()