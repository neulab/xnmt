import unittest

import dynet_config
dynet_config.set(random_seed=3)

import dynet as dy
import numpy
import random
import math

from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
import xnmt.batcher
from xnmt.input_reader import PlainTextReader
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculator import MLELoss
from xnmt.mlp import MLP
from xnmt.param_collection import ParamManager
from xnmt.translator import DefaultTranslator
from xnmt.loss_calculator import MLELoss
from xnmt.search_strategy import BeamSearch, GreedySearch
from xnmt.hyper_parameters import *
from xnmt.segmenting_encoder import *
from xnmt.segmenting_composer import *
from xnmt.constants import EPSILON
from xnmt.transducer import IdentitySeqTransducer
from scipy.stats import poisson

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
    self.loss_calculator = MLELoss()
    self.segmenting_encoder = SegmentingSeqTransducer(
      embed_encoder = self.segment_embed_encoder_bilstm,
      segment_composer =  self.segment_composer,
      final_transducer = BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      src_vocab = self.src_reader.vocab,
      trg_vocab = self.trg_reader.vocab,
    )

    self.model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=self.segmenting_encoder,
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(True)

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
      exp_length = self.src[batch_idx][i].original_length/length_prior
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
      res_enc = enc(res_embed)
      for i in range(len(self.src[batch_idx])):
        single_loss_test(batch_idx, i, res_embed, res_enc)

  def test_sample_softmax(self):
    enc = self.segmenting_encoder
    emb = self.inp_emb(0)
    # Sample from softmax during training
    self.model.set_train(True)
    enc(emb)
    self.assertEqual(enc.sample_action, SampleAction.SOFTMAX)
    # Argmax during Testing
    self.model.set_train(False)
    enc(emb)
    self.assertEqual(enc.sample_action, SampleAction.ARGMAX)
  
  def test_eps_greedy(self):
    enc = self.segmenting_encoder
    enc.eps = DefinedSequence([1.0])
    emb = self.inp_emb(0)
    self.model.set_train(True)
    enc(emb)
    self.assertEqual(enc.sample_action, SampleAction.LP)
    self.model.set_train(False)
    enc(emb)
    self.assertEqual(enc.sample_action, SampleAction.ARGMAX)

  def test_sample_from_poisson(self):
    enc = self.segmenting_encoder
    emb = self.inp_emb(0)
    enc.length_prior = 0.1
    enc.length_prior_alpha = DefinedSequence([1.0])
    results = enc.sample_from_poisson(encodings=[0 for _ in range(8)],
                                      batch_size=4)
    self.assertEqual([len(x) for x in results], [8 for _ in range(4)])


  def test_average_composer(self):
    enc = self.segmenting_encoder
    enc.segment_composer.encoder = IdentitySeqTransducer()
    enc.segment_composer.transformer = SumSegmentTransformer()
    enc(self.inp_emb(0))

if __name__ == "__main__":
  unittest.main()
