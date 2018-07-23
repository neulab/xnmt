import unittest

#import dynet_config
#dynet_config.set(random_seed=3)

import dynet as dy
import numpy
import random

from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import AutoRegressiveDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
import xnmt.batcher
from xnmt.input_reader import PlainTextReader, CharFromWordTextReader
from xnmt.lstm import UniLSTMSeqTransducer
from xnmt.translator import DefaultTranslator
from xnmt.loss_calculator import AutoRegressiveMLELoss
from xnmt.specialized_encoders.segmenting_encoder.segmenting_encoder import *
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import *
from xnmt.specialized_encoders.segmenting_encoder.length_prior import PoissonLengthPrior
from xnmt.specialized_encoders.segmenting_encoder.priors import PoissonPrior, GoldInputPrior
from xnmt.transform import AuxNonLinear, Linear
from xnmt.scorer import Softmax
from xnmt.transducer import IdentitySeqTransducer
from xnmt.vocab import Vocab
from xnmt.rl.policy_gradient import PolicyGradient
from xnmt.rl.eps_greedy import EpsilonGreedy
from xnmt.rl.confidence_penalty import ConfidencePenalty
from xnmt.test.utils import has_cython

class TestSegmentingEncoder(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 64
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.segment_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_composer = SumComposer()
    self.src_reader = CharFromWordTextReader()
    self.trg_reader = PlainTextReader()
    self.loss_calculator = AutoRegressiveMLELoss()


    baseline = Linear(input_dim=layer_dim, output_dim=1)
    policy_network = Linear(input_dim=layer_dim, output_dim=2)
    self.poisson_prior = PoissonPrior(mu=3.3)
    self.eps_greedy = EpsilonGreedy(eps_prob=0.0, prior=self.poisson_prior)
    self.conf_penalty = ConfidencePenalty()
    self.policy_gradient = PolicyGradient(input_dim=layer_dim,
                                          output_dim=2,
                                          baseline=baseline,
                                          policy_network=policy_network,
                                          z_normalization=True,
                                          conf_penalty=self.conf_penalty,
                                          sample=5)
    self.length_prior = PoissonLengthPrior(lmbd=3.3, weight=1)
    self.segmenting_encoder = SegmentingSeqTransducer(
      embed_encoder = self.segment_encoder_bilstm,
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      policy_learning = self.policy_gradient,
      eps_greedy = self.eps_greedy,
      length_prior = self.length_prior,
    )

    self.model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=self.segmenting_encoder,
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=100, input_dim=layer_dim),
                                    trg_embed_dim=layer_dim,
                                    bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(True)

    self.layer_dim = layer_dim
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    my_batcher = xnmt.batcher.TrgBatcher(batch_size=3, src_pad_token=1, trg_pad_token=2)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def test_reinforce_loss(self):
    self.model.global_fertility = 1.0
    loss = self.model.calc_loss(self.src[0], self.trg[0], AutoRegressiveMLELoss())
    reinforce_loss = self.model.calc_additional_loss(self.trg[0], self.model, loss)
    pl = self.model.encoder.policy_learning
    # Ensure correct length
    src = self.src[0]
    mask = src.mask.np_arr
    outputs = self.segmenting_encoder.compose_output
    actions = self.segmenting_encoder.segment_actions
    # Ensure sample == outputs
    self.assertEqual(len(outputs), pl.sample)
    self.assertEqual(len(actions), pl.sample)
    for sample_action in actions:
      for i, sample_item in enumerate(sample_action):
        # The last segmentation is 1
        self.assertEqual(sample_item[-1], src[i].len_unpadded())
        # Assert that all flagged actions are </s>
        list(self.assertEqual(pl.actions[j][0][i], 1) for j in range(len(mask[i])) if mask[i][j] == 1)
    self.assertTrue("mle" in loss.expr_factors)
    self.assertTrue("fertility" in loss.expr_factors)
    self.assertTrue("rl_reinf" in reinforce_loss.expr_factors)
    self.assertTrue("rl_baseline" in reinforce_loss.expr_factors)
    self.assertTrue("rl_confpen" in reinforce_loss.expr_factors)
    # Ensure we are sampling from the policy learning
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.POLICY)

  def calc_loss_single_batch(self):
    loss = self.model.calc_loss(self.src[0], self.trg[0], AutoRegressiveMLELoss())
    reinforce_loss = self.model.calc_additional_loss(self.trg[0], self.model, loss)
    return loss, reinforce_loss

  def test_gold_input(self):
    self.model.encoder.policy_learning = None
    self.model.encoder.eps_greedy = None
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.GOLD)

  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_sample_input(self):
    self.model.encoder.eps_greedy.eps_prob= 1.0
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.POLICY_SAMPLE)
    self.assertEqual(self.model.encoder.policy_learning.sampling_action, PolicyGradient.SamplingAction.PREDEFINED)

  def test_global_fertility(self):
    # Test Global fertility weight
    self.model.global_fertility = 1.0
    self.segmenting_encoder.policy_learning = None
    loss1, _ = self.calc_loss_single_batch()
    self.assertTrue("fertility" in loss1.expr_factors)
  
  def test_policy_train_test(self):
    self.model.set_train(True)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.policy_learning.sampling_action, PolicyGradient.SamplingAction.POLICY_CLP)
    self.model.set_train(False)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.policy_learning.sampling_action, PolicyGradient.SamplingAction.POLICY_AMAX)

  def test_no_policy_train_test(self):
    self.model.encoder.policy_learning = None
    self.model.set_train(True)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.PURE_SAMPLE)
    self.model.set_train(False)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.PURE_SAMPLE)

  def test_sample_during_search(self):
    self.model.set_train(False)
    self.model.encoder.sample_during_search = True
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.POLICY)

  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_policy_gold(self):
    self.model.encoder.eps_greedy.prior = GoldInputPrior("segment")
    self.model.encoder.eps_greedy.eps_prob = 1.0
    self.calc_loss_single_batch()

class TestComposing(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 64
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.segment_composer = SumComposer()
    self.src_reader = CharFromWordTextReader()
    self.trg_reader = PlainTextReader()
    self.loss_calculator = AutoRegressiveMLELoss()
    self.segmenting_encoder = SegmentingSeqTransducer(
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
    )

    self.model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=self.segmenting_encoder,
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=100, input_dim=layer_dim),
                                    trg_embed_dim=layer_dim,
                                    bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(True)

    self.layer_dim = layer_dim
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    my_batcher = xnmt.batcher.TrgBatcher(batch_size=3, src_pad_token=1, trg_pad_token=2)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def inp_emb(self, idx=0):
    self.model.start_sent(self.src[idx])
    embed = self.model.src_embedder.embed_sent(self.src[idx])
    return embed

  def test_lookup_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = LookupComposer(
        word_vocab = word_vocab,
        src_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))

  def test_charngram_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = CharNGramComposer(
        word_vocab = word_vocab,
        src_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))

  def test_add_multiple_segment_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = SumMultipleComposer(
      composers = [
        LookupComposer(word_vocab = word_vocab,
                                     src_vocab = self.src_reader.vocab,
                                     hidden_dim = self.layer_dim),
        CharNGramComposer(word_vocab = word_vocab,
                                 src_vocab = self.src_reader.vocab,
                                 hidden_dim = self.layer_dim)
      ]
    )
    enc.transduce(self.inp_emb(0))

  def test_sum_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = SumComposer()
    enc.transduce(self.inp_emb(0))

  def test_avg_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = AverageComposer()
    enc.transduce(self.inp_emb(0))

  def test_max_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = MaxComposer()
    enc.transduce(self.inp_emb(0))

  def test_convolution_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = ConvolutionComposer(ngram_size=1,
                                               embed_dim=self.layer_dim,
                                               hidden_dim=self.layer_dim)
    self.model.set_train(True)
    enc.transduce(self.inp_emb(0))
    enc.segment_composer = ConvolutionComposer(ngram_size=3,
                                               embed_dim=self.layer_dim,
                                               hidden_dim=self.layer_dim)
    self.model.set_train(True)
    enc.transduce(self.inp_emb(0))

  def test_transducer_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = SeqTransducerComposer(seq_transducer=BiLSTMSeqTransducer(input_dim=self.layer_dim,
                                                                                    hidden_dim=self.layer_dim))
    self.model.set_train(True)
    enc.transduce(self.inp_emb(0))

if __name__ == "__main__":
  unittest.main()
