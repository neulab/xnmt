import unittest

#import dynet_config
#dynet_config.set(random_seed=3)

import dynet as dy
import numpy
import random

from xnmt.modelparts.attenders import MlpAttender
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
import xnmt.events
from xnmt.eval import metrics
from xnmt import batchers, event_trigger
from xnmt.input_readers import PlainTextReader
from xnmt.input_readers import CharFromWordTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.models.translators import DefaultTranslator
from xnmt.loss_calculators import MLELoss, FeedbackLoss, GlobalFertilityLoss, CompositeLoss
from xnmt.specialized_encoders.segmenting_encoder.segmenting_encoder import SegmentingSeqTransducer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import SumComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import SumMultipleComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import AverageComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import ConvolutionComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import CharNGramComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import SeqTransducerComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import MaxComposer
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import LookupComposer
from xnmt.specialized_encoders.segmenting_encoder.reporter import SegmentPLLogger
from xnmt.specialized_encoders.segmenting_encoder.length_prior import PoissonLengthPrior
from xnmt.specialized_encoders.segmenting_encoder.priors import PoissonPrior, GoldInputPrior
from xnmt.param_collections import ParamManager
from xnmt.modelparts.transforms import AuxNonLinear, Linear
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab
from xnmt.rl.policy_gradient import PolicyGradient
from xnmt.rl.eps_greedy import EpsilonGreedy
from xnmt.rl.confidence_penalty import ConfidencePenalty
from xnmt.utils import has_cython

class TestSegmentingEncoder(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 4
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.segment_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_composer = SumComposer()

    self.src_reader = CharFromWordTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.charvocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.en.vocab"))
    self.loss_calculator = FeedbackLoss(child_loss=MLELoss(), repeat=5)

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
                                          conf_penalty=self.conf_penalty)
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
    event_trigger.set_train(True)

    self.layer_dim = layer_dim
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def test_reinforce_loss(self):
    fertility_loss = GlobalFertilityLoss()
    mle_loss = MLELoss()
    loss = CompositeLoss(pt_losses=[mle_loss, fertility_loss]).calc_loss(self.model, self.src[0], self.trg[0])
    reinforce_loss = event_trigger.calc_additional_loss(self.trg[0], self.model, loss)
    pl = self.model.encoder.policy_learning
    # Ensure correct length
    src = self.src[0]
    mask = src.mask.np_arr
    outputs = self.segmenting_encoder.compose_output
    actions = self.segmenting_encoder.segment_actions
    # Ensure sample == outputs
    for i, sample_item in enumerate(actions):
      # The last segmentation is 1
      self.assertEqual(sample_item[-1], src[i].len_unpadded())
    self.assertTrue("mle" in loss.expr_factors)
    self.assertTrue("global_fertility" in loss.expr_factors)
    self.assertTrue("rl_reinf" in reinforce_loss.expr_factors)
    self.assertTrue("rl_baseline" in reinforce_loss.expr_factors)
    self.assertTrue("rl_confpen" in reinforce_loss.expr_factors)
    # Ensure we are sampling from the policy learning
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.POLICY)

  def calc_loss_single_batch(self):
    loss = MLELoss().calc_loss(self.model, self.src[0], self.trg[0])
    reinforce_loss = event_trigger.calc_additional_loss(self.trg[0], self.model, loss)
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
  
  def test_policy_train_test(self):
    event_trigger.set_train(True)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.policy_learning.sampling_action, PolicyGradient.SamplingAction.POLICY_CLP)
    event_trigger.set_train(False)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.policy_learning.sampling_action, PolicyGradient.SamplingAction.POLICY_AMAX)

  def test_no_policy_train_test(self):
    self.model.encoder.policy_learning = None
    event_trigger.set_train(True)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.PURE_SAMPLE)
    event_trigger.set_train(False)
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.PURE_SAMPLE)

  def test_sample_during_search(self):
    event_trigger.set_train(False)
    self.model.encoder.sample_during_search = True
    self.calc_loss_single_batch()
    self.assertEqual(self.model.encoder.segmenting_action, SegmentingSeqTransducer.SegmentingAction.POLICY)

  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_policy_gold(self):
    self.model.encoder.eps_greedy.prior = GoldInputPrior("segment")
    self.model.encoder.eps_greedy.eps_prob = 1.0
    self.calc_loss_single_batch()

  def test_reporter(self):
    self.model.encoder.reporter = SegmentPLLogger("test/tmp/seg-report.log", self.model.src_reader.vocab)
    self.calc_loss_single_batch()

class TestComposing(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 4
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.segment_composer = SumComposer()
    self.src_reader = CharFromWordTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.charvocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.en.vocab"))
    self.loss_calculator = FeedbackLoss(child_loss=MLELoss(), repeat=5)
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
    event_trigger.set_train(True)

    self.layer_dim = layer_dim
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def inp_emb(self, idx=0):
    event_trigger.start_sent(self.src[idx])
    embed = self.model.src_embedder.embed_sent(self.src[idx])
    return embed

  def test_lookup_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    enc.segment_composer = LookupComposer(
        word_vocab = word_vocab,
        char_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))

  def test_charngram_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    enc.segment_composer = CharNGramComposer(
        word_vocab = word_vocab,
        char_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))

  def test_lookup_composer_learn(self):
    enc = self.segmenting_encoder
    char_vocab = Vocab(i2w=['a', 'b', 'c', 'd'])
    enc.segment_composer = LookupComposer(
        word_vocab = None,
        char_vocab = char_vocab,
        hidden_dim = self.layer_dim,
        vocab_size = 4
    )
    event_trigger.set_train(True)
    enc.segment_composer.set_word((0, 1, 2)) # abc 0
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 2, 1)) # acb 1
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 3, 2)) # adc 2
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 1, 2)) # abc 0
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((1, 3, 2)) # bdc 3
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((3, 3, 3)) # ddd 1 -> acb is the oldest
    enc.segment_composer.transduce([])
    act = dict(enc.segment_composer.lrucache.items())
    exp = {'abc': 0, 'ddd': 1, 'adc': 2, 'bdc': 3}
    self.assertDictEqual(act, exp)
    
    enc.segment_composer.set_word((0, 2, 1))
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 3, 2))
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 1, 2))  # abc 0
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((1, 3, 2))  # bdc 3
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((3, 3, 3))
    enc.segment_composer.transduce([])
    enc.segment_composer.set_word((0, 3, 1))
    enc.segment_composer.transduce([])

    event_trigger.set_train(False)
    enc.segment_composer.set_word((3, 3, 2))
    enc.segment_composer.transduce([])

  def test_chargram_composer_learn(self):
    enc = self.segmenting_encoder
    char_vocab = Vocab(i2w=['a', 'b', 'c', 'd'])
    enc.segment_composer = CharNGramComposer(
        word_vocab = None,
        char_vocab = char_vocab,
        hidden_dim = self.layer_dim,
        ngram_size = 2,
        vocab_size = 5,
    )
    event_trigger.set_train(True)
    enc.segment_composer.set_word((0, 1, 2)) # a:0, ab:1, b: 2, bc: 3, c: 4
    enc.segment_composer.transduce([])
    act = dict(enc.segment_composer.lrucache.items())
    exp = {'a': 0, 'ab': 1, 'b': 2, 'bc': 3, 'c': 4}
    self.assertDictEqual(act, exp)

    enc.segment_composer.set_word((2, 3)) # c, cd, d
    enc.segment_composer.transduce([])
    act = dict(enc.segment_composer.lrucache.items())
    exp = {'cd': 0, 'd': 1, 'b': 2, 'bc': 3, 'c': 4}
    self.assertDictEqual(act, exp)

  def test_add_multiple_segment_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    enc.segment_composer = SumMultipleComposer(
      composers = [
        LookupComposer(word_vocab = word_vocab,
                       char_vocab = self.src_reader.vocab,
                       hidden_dim = self.layer_dim),
        CharNGramComposer(word_vocab = word_vocab,
                          char_vocab = self.src_reader.vocab,
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
    event_trigger.set_train(True)
    enc.transduce(self.inp_emb(0))
    enc.segment_composer = ConvolutionComposer(ngram_size=3,
                                               embed_dim=self.layer_dim,
                                               hidden_dim=self.layer_dim)
    event_trigger.set_train(True)
    enc.transduce(self.inp_emb(0))

  def test_transducer_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = SeqTransducerComposer(seq_transducer=BiLSTMSeqTransducer(input_dim=self.layer_dim,
                                                                                    hidden_dim=self.layer_dim))
    event_trigger.set_train(True)
    enc.transduce(self.inp_emb(0))

class TestSegmentationFMeasureEvaluator(unittest.TestCase):
  def test_fmeasure(self):
    self.assertEqual(metrics.SegmentationFMeasureEvaluator().evaluate_one_sent("ab c def".split(), "a bc def".split()).value(),
                     0.80)

  def test_fmeasure_error(self):
    with self.assertRaises(Exception) as context:
      metrics.SegmentationFMeasureEvaluator().evaluate_one_sent("aaa b".split(), "aaa".split())

if __name__ == "__main__":
  unittest.main()
