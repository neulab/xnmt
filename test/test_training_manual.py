"""
Manually initialized toy model, test for both DyNet and Pytorch backends to produce exactly the same loss value.
"""

import unittest

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt.param_initializers import NumpyInitializer, InitializerSequence
from xnmt.modelparts.attenders import DotAttender
from xnmt.batchers import SrcBatcher
from xnmt.models.classifiers import SequenceClassifier
from xnmt.modelparts.bridges import NoBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
import xnmt.events
from xnmt.input_readers import PlainTextReader, IDReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculators import MLELoss
from xnmt.loss_trackers import TrainLossTracker
from xnmt import optimizers
from xnmt.param_collections import ParamManager
from xnmt.train import regimens
from xnmt.modelparts.transforms import NonLinear
from xnmt.models.translators.default import DefaultTranslator
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab



class TestManualBasicSeq2seq(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def run_training(self, num_layers, bi_encoder, epochs, lr):
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/ab-ba.txt"
    train_args['trg_file'] = "test/data/ab-ba.txt"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    emb_arr_5_2 = np.asarray([[-0.1, 0.1],[-0.2, 0.2],[-0.3, 0.3],[-0.4, 0.4],[-0.5, 0.5],])
    proj_arr_2_4 = np.asarray([
      [-0.1, -0.2, -0.3, -0.4],
      [0.1, 0.2, 0.3, 0.4],
    ])
    lstm_arr_8_2 = np.asarray([
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
    ])
    lstm_arr_4_2 = np.asarray([
      [-0.1, -0.2],
      [-0.1, -0.2],
      [-0.1, -0.2],
      [-0.1, -0.2],
    ])
    lstm_arr_4_1 = np.asarray([
      [-0.1],
      [-0.1],
      [-0.1],
      [-0.1],
    ])
    dec_lstm_arr_8_4 = np.asarray([
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
    ])
    if bi_encoder:
      assert num_layers==1
      encoder = BiLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    param_init=InitializerSequence([InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),  # fwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),  # bwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)])]  # bwd_l0_hh
                                    ),
                                    layers=num_layers)
    else:
      encoder = UniLSTMSeqTransducer(input_dim=layer_dim,
                                     hidden_dim=layer_dim,
                                     param_init=NumpyInitializer(lstm_arr_8_2),
                                     layers=num_layers)
    train_args['model'] = \
      DefaultTranslator(
        src_reader=PlainTextReader(vocab=vocab),
        trg_reader=PlainTextReader(vocab=vocab),
        src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(emb_arr_5_2)),
        encoder=encoder,
        attender=DotAttender(),
        decoder=AutoRegressiveDecoder(
          input_dim=layer_dim,
          embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(emb_arr_5_2)),
          rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                   hidden_dim=layer_dim,
                                   decoder_input_dim=layer_dim,
                                   layers=num_layers,
                                   param_init=InitializerSequence(
                                     [NumpyInitializer(dec_lstm_arr_8_4)] + [NumpyInitializer(lstm_arr_8_2)] * (num_layers*2-1)),
                                   yaml_path="model.decoder.rnn"),
          transform=NonLinear(input_dim=layer_dim * 2, output_dim=layer_dim, param_init=NumpyInitializer(proj_arr_2_4)),
          scorer=Softmax(input_dim=layer_dim, vocab_size=vocab_size ,param_init=NumpyInitializer(emb_arr_5_2)),
          bridge=NoBridge(dec_dim=layer_dim, dec_layers=num_layers)),
      )
    train_args['dev_tasks'] = []
    train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = epochs
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  def assert_loss_value(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    self.assertAlmostEqual(training_regimen.train_loss_tracker.epoch_loss.sum_factors(), val, places=places)

  def assert_trained_emb_params(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    if xnmt.backend_dynet:
      trained_src_emb = training_regimen.model.src_embedder.embeddings.as_array()
    else:
      trained_src_emb = tt.npvalue(training_regimen.model.src_embedder.embeddings._parameters['weight'].data)
      val = val.T
    np.testing.assert_almost_equal(trained_src_emb, val, decimal=places)

  def test_loss_basic(self):
    self.assert_loss_value(9.657152, places=5, num_layers=1, bi_encoder=False, epochs=1, lr=0.1)

  # def test_loss_two_epochs(self):
  #   self.assert_loss_value(6.585153, places=2, num_layers=1, bi_encoder=False, epochs=2, lr=10)
  #
  # def test_emb_weights_two_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.19995555, 0.19998002], [-0.2998707, 0.30003682], [-0.39986575, 0.40003762], [-0.5, 0.5]])
  #   self.assert_trained_emb_params(expected, places=5, num_layers=1, bi_encoder=False, epochs=2, lr=10)

  def test_loss_two_layers(self):
    self.assert_loss_value(9.656650, places=5, num_layers=2, bi_encoder=False, epochs=1, lr=0.1)

  def test_loss_bidirectional(self):
    self.assert_loss_value(9.657083, places=5, num_layers=1, bi_encoder=True, epochs=1, lr=0.1)


class TestManualClassifier(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def run_training(self, num_layers, bi_encoder, epochs, lr):
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/ab-ba.txt"
    train_args['trg_file'] = "test/data/ab-ba.lbl"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    emb_arr_5_2 = np.asarray([[-0.1, 0.1],[-0.2, 0.2],[-0.3, 0.3],[-0.4, 0.4],[-0.5, 0.5],])
    out_arr_2_2 = np.asarray([[-0.1, 0.1],[-0.2, 0.2],])
    proj_arr_2_2 = np.asarray([
      [-0.1, -0.2],
      [0.1, 0.2],
    ])
    lstm_arr_8_2 = np.asarray([
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
    ])
    lstm_arr_4_2 = np.asarray([
      [-0.1, -0.2],
      [-0.1, -0.2],
      [-0.1, -0.2],
      [-0.1, -0.2],
    ])
    lstm_arr_4_1 = np.asarray([
      [-0.1],
      [-0.1],
      [-0.1],
      [-0.1],
    ])
    if bi_encoder:
      assert num_layers==1
      encoder = BiLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    param_init=InitializerSequence([InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),  # fwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),  # bwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)])]  # bwd_l0_hh
                                    ),
                                    layers=num_layers)
    else:
      encoder = UniLSTMSeqTransducer(input_dim=layer_dim,
                                     hidden_dim=layer_dim,
                                     param_init=NumpyInitializer(lstm_arr_8_2),
                                     layers=num_layers)
    train_args['model'] = \
      SequenceClassifier(
        src_reader=PlainTextReader(vocab=vocab),
        trg_reader=IDReader(),
        src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(emb_arr_5_2)),
        encoder=encoder,
        transform=NonLinear(input_dim=layer_dim, output_dim=layer_dim, param_init=NumpyInitializer(proj_arr_2_2)),
        scorer=Softmax(input_dim=layer_dim, vocab_size=2 ,param_init=NumpyInitializer(out_arr_2_2)),
      )
    train_args['dev_tasks'] = []
    train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = epochs
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  def assert_loss_value(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    self.assertAlmostEqual(training_regimen.train_loss_tracker.epoch_loss.sum_factors(), val, places=places)

  def assert_trained_emb_params(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    if xnmt.backend_dynet:
      trained_src_emb = training_regimen.model.src_embedder.embeddings.as_array()
    else:
      trained_src_emb = tt.npvalue(training_regimen.model.src_embedder.embeddings._parameters['weight'].data)
      val = val.T
    np.testing.assert_almost_equal(trained_src_emb, val, decimal=places)

  def test_loss_basic(self):
    self.assert_loss_value(1.386299, places=5, num_layers=1, bi_encoder=False, epochs=1, lr=0.1)

  def test_loss_twolayer(self):
    self.assert_loss_value(1.386294, places=5, num_layers=2, bi_encoder=False, epochs=1, lr=0.1)

  def test__loss_bidirectional(self):
    self.assert_loss_value(1.386302, places=5, num_layers=1, bi_encoder=True, epochs=1, lr=0.1)

  def test_loss_two_epochs(self):
    self.assert_loss_value(1.386299, places=5, num_layers=1, bi_encoder=False, epochs=2, lr=0.1)

  def test_loss_five_epochs(self):
    self.assert_loss_value(1.386299, places=2, num_layers=1, bi_encoder=False, epochs=5, lr=0.1)

  def test_emb_weights_two_epochs(self):
    expected = np.asarray(
      [[-0.1, 0.1], [-0.2000126, 0.1999742], [-0.2989355, 0.302096], [-0.40107998, 0.39787248], [-0.5, 0.5]])
    self.assert_trained_emb_params(expected, places=5, num_layers=1, bi_encoder=False, epochs=2, lr=10)

  def test_emb_weights_five_epochs(self):
    expected = np.asarray(
      [[-0.1, 0.1], [-0.20250981, 0.19391325], [-0.29897961, 0.30119216], [-0.40397269, 0.39145479], [-0.5, 0.5]])
    self.assert_trained_emb_params(expected, places=3, num_layers=1, bi_encoder=False, epochs=5, lr=10)


if __name__ == '__main__':
  unittest.main()
