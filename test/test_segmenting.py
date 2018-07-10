import unittest

import dynet_config
dynet_config.set(random_seed=3)

import dynet as dy
import numpy
import random
import math
from scipy.stats import poisson

from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import AutoRegressiveDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
import xnmt.batcher
from xnmt.input_reader import PlainTextReader, CharFromWordTextReader
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculator import AutoRegressiveMLELoss
from xnmt.param_collection import ParamManager
from xnmt.translator import DefaultTranslator
from xnmt.loss_calculator import AutoRegressiveMLELoss
from xnmt.search_strategy import BeamSearch, GreedySearch
from xnmt.hyper_parameters import *
from xnmt.specialized_encoders.segmenting_encoder.segmenting_encoder import *
from xnmt.specialized_encoders.segmenting_encoder.segmenting_composer import *
from xnmt.transform import AuxNonLinear, Linear
from xnmt.scorer import Softmax
from xnmt.constants import EPSILON
from xnmt.transducer import IdentitySeqTransducer
from xnmt.vocab import Vocab
from xnmt.rl.policy_gradient import PolicyGradient
from xnmt.rl.eps_greedy import EpsilonGreedy
from xnmt.priors import PoissonPrior

class TestSegmentingEncoder(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 64
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.tail_transformer = TailSegmentTransformer()
    self.segment_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_embed_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_composer = SegmentComposer(encoder=self.segment_encoder_bilstm,
                                            transformer=self.tail_transformer)
    self.src_reader = PlainTextReader()
    self.trg_reader = PlainTextReader()
    self.loss_calculator = AutoRegressiveMLELoss()
    self.segmenting_encoder = SegmentingSeqTransducer(
      embed_encoder = self.segment_embed_encoder_bilstm,
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      src_vocab = self.src_reader.vocab,
      trg_vocab = self.trg_reader.vocab,
      embed_encoder_dim = layer_dim,
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

  def extract_mask(self, mask, i, col_size):
    if mask is None:
      return numpy.zeros(col_size)
    else:
      return mask.np_arr[i]

  def test_calc_loss(self):
    enc = self.segmenting_encoder
    enc.length_prior = 3.3
    enc.length_prior_alpha = DefinedSequence([1.0])
    
    # Outputs, priors
    def single_loss_test(batch_idx, i, res_embed, res_enc):
      seg_dec = enc.segment_decisions[i]
      length_prior = enc.length_prior
      # mask from the actual encoding
      res_mask = self.extract_mask(res_enc.mask, i, len(seg_dec))
      # mask from the encoding of tokens
      enc_mask = self.extract_mask(res_embed.mask, i, len(self.src[batch_idx][i]))
      loss_res_mask = enc.enc_mask[i] # mask used to calculate additional loss
      # expected
      exp_length = self.src[batch_idx][i].len_unpadded()/length_prior
      exp_lp = math.log(poisson.pmf(len(seg_dec), exp_length))
      exp_flag_enc = len(res_mask) - numpy.count_nonzero(res_mask)
      exp_loss_res_mask = numpy.count_nonzero(1-enc_mask) - 1
      # actual
      act_length = enc.expected_length[i]
      act_lp = dy.pick_batch_elem(enc.segment_length_prior, i).value()
      act_flag_enc = len(seg_dec)
      act_loss_res_mask = numpy.count_nonzero(loss_res_mask)
      # Assertions
      self.assertAlmostEqual(exp_length, act_length)
      self.assertAlmostEqual(exp_lp, act_lp, places=6)
      self.assertEqual(exp_flag_enc, act_flag_enc)
      self.assertEqual(exp_loss_res_mask, exp_loss_res_mask)

    # Test For every items in the batch
    for batch_idx in range(len(self.src)):
      res_embed = self.inp_emb(batch_idx)
      res_enc = enc.transduce(res_embed)
      for i in range(len(self.src[batch_idx])):
        single_loss_test(batch_idx, i, res_embed, res_enc)

  def test_sample_softmax(self):
    enc = self.segmenting_encoder
    emb = self.inp_emb(0)
    # Sample from softmax during training
    self.model.set_train(True)
    enc.transduce(emb)
    self.assertEqual(enc.sample_action, SampleAction.SOFTMAX)
    # Argmax during Testing
    self.model.set_train(False)
    enc.transduce(emb)
    self.assertEqual(enc.sample_action, SampleAction.ARGMAX)
  
  def test_eps_greedy(self):
    enc = self.segmenting_encoder
    enc.eps = DefinedSequence([1.0])
    emb = self.inp_emb(0)
    self.model.set_train(True)
    enc.transduce(emb)
    self.assertEqual(enc.sample_action, SampleAction.LP)
    self.model.set_train(False)
    enc.transduce(emb)
    self.assertEqual(enc.sample_action, SampleAction.ARGMAX)

  def test_sample_from_poisson(self):
    enc = self.segmenting_encoder
    emb = self.inp_emb(0)
    enc.length_prior = 0.1
    enc.length_prior_alpha = DefinedSequence([1.0])
    results = enc.sample_from_poisson(encodings=[0 for _ in range(8)],
                                      batch_size=4)
    self.assertEqual([len(x) for x in results], [8 for _ in range(4)])

  def test_compose_char(self):
    enc = self.segmenting_encoder
    enc.embed_encoder = IdentitySeqTransducer()
    enc.compose_char = True
    enc.transduce(self.inp_emb(0))

  def test_sum_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer.encoder = IdentitySeqTransducer()
    enc.segment_composer.transformer = SumSegmentTransformer()
    enc.transduce(self.inp_emb(0))

  def test_avg_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer.encoder = IdentitySeqTransducer()
    enc.segment_composer.transformer = AverageSegmentTransformer()
    enc.transduce(self.inp_emb(0))

  def test_max_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer.encoder = IdentitySeqTransducer()
    enc.segment_composer.transformer = MaxSegmentTransformer()
    enc.transduce(self.inp_emb(0))

  def test_convolution_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer = ConvolutionSegmentComposer(ngram_size=3,
                                                      dropout=0.5,
                                                      embed_dim=self.layer_dim,
                                                      hidden_dim=self.layer_dim)
    self.model.set_train(True)
    enc.transduce(self.inp_emb(0))

class TestPriorSegmentation(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 64
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.tail_transformer = TailSegmentTransformer()
    self.segment_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_embed_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_composer = SegmentComposer(encoder=self.segment_encoder_bilstm,
                                            transformer=self.tail_transformer)
    self.src_reader = CharFromWordTextReader()
    self.trg_reader = PlainTextReader()
    self.loss_calculator = AutoRegressiveMLELoss()
    self.segmenting_encoder = SegmentingSeqTransducer(
      embed_encoder = self.segment_embed_encoder_bilstm,
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      src_vocab = self.src_reader.vocab,
      trg_vocab = self.trg_reader.vocab,
      embed_encoder_dim = layer_dim,
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

  def test_embed_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = WordEmbeddingSegmentComposer(
        word_vocab = word_vocab,
        src_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))
    last_sent = self.src[0][-1]
    last_converted_word = self.src_reader.vocab[last_sent[last_sent.len_unpadded()]]
    assert enc.segment_composer.word == last_converted_word

  def test_charngram_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = CharNGramSegmentComposer(
        word_vocab = word_vocab,
        src_vocab = self.src_reader.vocab,
        hidden_dim = self.layer_dim
    )
    enc.transduce(self.inp_emb(0))  
  
  def test_add_multiple_segment_composer(self):
    enc = self.segmenting_encoder
    word_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    word_vocab.freeze()
    enc.segment_composer = SumMultipleSegmentComposer(
      segment_composers = [
        WordEmbeddingSegmentComposer(word_vocab = word_vocab,
                                     src_vocab = self.src_reader.vocab,
                                     hidden_dim = self.layer_dim),
        CharNGramSegmentComposer(word_vocab = word_vocab,
                                 src_vocab = self.src_reader.vocab,
                                 hidden_dim = self.layer_dim)
      ]
    )
    enc.transduce(self.inp_emb(0))

class TestSegmentingEncoderTraining(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 64
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.tail_transformer = TailSegmentTransformer()
    self.segment_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_embed_encoder_bilstm = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)
    self.segment_composer = SegmentComposer(encoder=self.segment_encoder_bilstm,
                                            transformer=self.tail_transformer)
    self.src_reader = CharFromWordTextReader()
    self.trg_reader = PlainTextReader()
    self.loss_calculator = AutoRegressiveMLELoss()


    baseline = Linear(input_dim=layer_dim, output_dim=layer_dim)
    policy_network = Linear(input_dim=layer_dim, output_dim=2)
    
    self.policy_gradient = PolicyGradient(baseline=baseline,
                                          policy_network=policy_network,
                                          z_normalization=True,
                                          sample=2)

    self.poisson_prior = PoissonPrior(mu=3.3)
    self.length_prior = LengthPrior(prior=self.poisson_prior, weight=1)
    self.eps_greedy = EpsilonGreedy(eps_prob=0.0, prior=self.poisson_prior)

    self.segmenting_encoder = SegmentingSeqTransducer(
      embed_encoder = self.segment_embed_encoder_bilstm,
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      src_vocab = self.src_reader.vocab,
      trg_vocab = self.trg_reader.vocab,
      embed_encoder_dim = layer_dim,
      policy_learning = self.policy_gradient,
      eps_greedy = self.eps_greedy,
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

  def test_global_fertility(self):
    # Test Global fertility weight
    self.model.global_fertility = 1.0
    self.segmenting_encoder.policy_learning = None
    loss1 = self.model.calc_loss(self.src[0], self.trg[0], AutoRegressiveMLELoss())
    self.model.global_fertility = 0.5
    loss2 = self.model.calc_loss(self.src[0], self.trg[0], AutoRegressiveMLELoss())
    numpy.testing.assert_almost_equal(loss1["fertility"].npvalue()/2, loss2["fertility"].npvalue())

if __name__ == "__main__":
  unittest.main()
