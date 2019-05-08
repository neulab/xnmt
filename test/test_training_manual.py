"""
Manually initialized toy model, test for both DyNet and Pytorch backends to produce exactly the same loss value.
"""

import unittest

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt import event_trigger
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


class ManualTestingBaseClass(object):

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

  def assert_trained_emb_grads(self, val, places, epochs=1, *args, **kwargs):
    training_regimen = self.run_training(epochs=epochs-1, *args, **kwargs)
    # last epoch is done manually and without calling update():
    src, trg = next(training_regimen.next_minibatch())
    tt.reset_graph()
    event_trigger.set_train(True)
    loss_builder = training_regimen.training_step(src, trg)
    loss = loss_builder.compute(comb_method=training_regimen.loss_comb_method)
    training_regimen.backward(loss)
    # importantly: no update() here because that would zero out the dynet gradients

    if xnmt.backend_dynet:
      actual_grads = training_regimen.model.src_embedder.embeddings.grad_as_array()
    else:
      actual_grads = tt.npvalue(training_regimen.model.src_embedder.embeddings._parameters['weight'].grad)
      val = val.T
    np.testing.assert_almost_equal(actual_grads, val, decimal=places)


class TestManualBasicSeq2seq(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def assert_trained_seq2seq_params(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(num_layers=1, bi_encoder=False, *args, **kwargs)
    if xnmt.backend_dynet:
      trained_src_emb = training_regimen.model.src_embedder.embeddings.as_array()
      trained_encoder = training_regimen.model.encoder.Wx[0].as_array(), \
                        training_regimen.model.encoder.Wh[0].as_array(), \
                        training_regimen.model.encoder.b[0].as_array()
      trained_decoder = training_regimen.model.decoder.rnn.Wx[0].as_array(), \
                        training_regimen.model.decoder.rnn.Wh[0].as_array(), \
                        training_regimen.model.decoder.rnn.b[0].as_array()
      trained_trg_emb = training_regimen.model.decoder.embedder.embeddings.as_array()
      trained_transform = training_regimen.model.decoder.transform.W1.as_array(), \
                          training_regimen.model.decoder.transform.b1.as_array()
      trained_out = training_regimen.model.decoder.scorer.output_projector.W1.as_array(), \
                    training_regimen.model.decoder.scorer.output_projector.b1.as_array()
    else:
      trained_src_emb = tt.npvalue(training_regimen.model.src_embedder.embeddings._parameters['weight'].data)
      trained_trg_emb = tt.npvalue(training_regimen.model.decoder.embedder.embeddings._parameters['weight'].data)
      trained_transform = tt.npvalue(training_regimen.model.decoder.transform.linear._parameters['weight'].data), \
                          tt.npvalue(training_regimen.model.decoder.transform.linear._parameters['bias'].data)
      trained_out= tt.npvalue(training_regimen.model.decoder.scorer.output_projector.linear._parameters['weight'].data), \
                   tt.npvalue(training_regimen.model.decoder.scorer.output_projector.linear._parameters['bias'].data)
      trained_encoder= tt.npvalue(training_regimen.model.encoder.layers[0]._parameters['weight_ih'].data), \
                       tt.npvalue(training_regimen.model.encoder.layers[0]._parameters['weight_hh'].data), \
                       tt.npvalue(training_regimen.model.encoder.layers[0]._parameters['bias_ih'].data)
      h_dim = trained_encoder[0].shape[1]//4
      # change ifgo -> ifog; subtract 1-initialized forget gates
      trained_encoder = np.concatenate([trained_encoder[0][:, :h_dim*2], trained_encoder[0][:, h_dim*3:], trained_encoder[0][:, h_dim*2:h_dim*3]], axis=1), \
                        np.concatenate([trained_encoder[1][:, :h_dim*2], trained_encoder[1][:, h_dim*3:], trained_encoder[1][:, h_dim*2:h_dim * 3]], axis=1), \
                        np.concatenate([trained_encoder[2][:h_dim], trained_encoder[2][h_dim:h_dim*2]-1, trained_encoder[2][h_dim*3:], trained_encoder[2][h_dim*2:h_dim*3]], axis=0),
      trained_decoder = tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_ih'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_hh'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['bias_ih'].data)
      h_dim = trained_decoder[0].shape[1]//4
      trained_decoder = np.concatenate([trained_decoder[0][:, :h_dim*2], trained_decoder[0][:, h_dim*3:], trained_decoder[0][:, h_dim*2:h_dim*3]], axis=1), \
                        np.concatenate([trained_decoder[1][:, :h_dim*2], trained_decoder[1][:, h_dim*3:], trained_decoder[1][:, h_dim*2:h_dim * 3]], axis=1), \
                        np.concatenate([trained_decoder[2][:h_dim], trained_decoder[2][h_dim:h_dim*2]-1, trained_decoder[2][h_dim*3:], trained_decoder[2][h_dim*2:h_dim*3]], axis=0),
      for k,v in val.items():
        if type(v)==tuple: val[k] = tuple(vi.T for vi in v)
        else: val[k] = v.T
    np.testing.assert_almost_equal(trained_src_emb, val['src_emb'], decimal=places)
    np.testing.assert_almost_equal(trained_trg_emb, val['trg_emb'], decimal=places)
    np.testing.assert_almost_equal(trained_transform[0], val['transform'][0], decimal=places)
    np.testing.assert_almost_equal(trained_transform[1], val['transform'][1], decimal=places)
    np.testing.assert_almost_equal(trained_out[0], val['out'][0], decimal=places)
    np.testing.assert_almost_equal(trained_out[1], val['out'][1], decimal=places)
    np.testing.assert_almost_equal(trained_encoder[0], val['encoder'][0], decimal=places)
    np.testing.assert_almost_equal(trained_encoder[1], val['encoder'][1], decimal=places)
    np.testing.assert_almost_equal(trained_encoder[2], val['encoder'][2], decimal=places)
    np.testing.assert_almost_equal(trained_decoder[0], val['decoder'][0], decimal=places)
    np.testing.assert_almost_equal(trained_decoder[1], val['decoder'][1], decimal=places)
    np.testing.assert_almost_equal(trained_decoder[2], val['decoder'][2], decimal=places)

  def run_training(self, num_layers=1, bi_encoder=False, epochs=1, lr=0.1):
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
    # note: dynet uses i|f|o|g, while pytorch uses i|f|g|o order; let's make sure to initialize output and update matrices to the same value
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
                                                                     NumpyInitializer(lstm_arr_4_2),   # fwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),   # bwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)])] # bwd_l0_hh
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
    if xnmt.backend_dynet:
      train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=-1)
    else:
      train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=0, rescale_grads=False)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = epochs
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  ####### OK #########
  # def test_loss_basic(self):
  #   self.assert_loss_value(9.657152, places=5)
  #
  # def test_loss_two_epochs(self):
  #   self.assert_loss_value(6.585153, places=2, epochs=2, lr=10)
  #
  # def test_loss_two_layers(self):
  #   self.assert_loss_value(9.656650, places=5, num_layers=2)
  #
  # def test_loss_bidirectional(self):
  #   self.assert_loss_value(9.657083, places=5, bi_encoder=True)
  #
  # def test_emb_weights_one_epoch(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20184304, 0.19631392], [-0.30349943, 0.29300117], [-0.40391687, 0.39216626], [-0.5, 0.5]])
  #   self.assert_trained_emb_params(expected, places=5, lr=100)
  #
  # def test_emb_grads_one_epoch(self):
  #   expected = np.asarray(
  #     [[0, 0], [1.84304245e-5, 3.68608489e-5], [3.49941438e-5, 6.99882876e-5], [3.91686735e-5, 7.83373471e-5], [0, 0]])
  #   self.assert_trained_emb_grads(expected, places=9, lr=10)
  #

  ##### TODO ###########

  def test_all_grads_one_epoch(self):
    # TODO: check if ALL params are equal after one epoch (so far, only checked the embedding params)
    #
    expected = {
      'src_emb': np.asarray([[-0.1, 0.1 ], [-0.2001843, 0.19963139], [-0.30034995, 0.29930013], [-0.4003917, 0.39921662], [-0.5, 0.5]]),
      'trg_emb': np.asarray([[-0.09716769, 0.10566463], [-0.2, 0.2], [-0.30496135, 0.29007733], [-0.41122547, 0.37754905], [-0.5, 0.5 ]]),
      'transform': (np.asarray([[-0.12163746, -0.17607024, -0.30009004, -0.39990318], [ 0.12163746, 0.17607024, 0.30009004, 0.39990318]]),
                    np.asarray([ 0.00683933, -0.00683933])),
      'out': (np.asarray([[-0.06577028, 0.06577028], [-0.23925397, 0.23925397], [-0.31620809, 0.31620809], [-0.41308054, 0.41308054], [-0.46568716, 0.46568716]]),
              np.asarray([-11.9862957, 8.00685501, 8.00000381, 7.99314737, -12.01371288])),
      'encoder': (np.asarray([[-0.10004691, -0.19995309], [ 0.10009123, 0.19990878], [-0.10001086, -0.19998914], [ 0.10002112, 0.19997889], [-0.1000402, -0.19995981], [ 0.10007835, 0.19992165], [-0.09733033, -0.20266968], [ 0.10536621,  0.1946338 ]]),
                  np.asarray([[-0.1000007, -0.19999924], [ 0.10000135, 0.19999854], [-0.10000048, -0.19999948], [ 0.10000093, 0.19999899], [-0.10000127, -0.19999863], [ 0.10000247, 0.19999734], [-0.09995046, -0.20005347], [ 0.10009786, 0.19989437]]),
                  np.asarray([ 1.38808842e-04, -2.69481330e-04, 4.07925472e-05, -7.95210435e-05, 1.38590593e-04, -2.70244025e-04, -8.30172468e-03, -1.66236106e-02])),
      'decoder': (np.asarray([[-0.10089689, -0.19910312, -0.10003209, -0.19996549], [ 0.10188823, 0.19811177, 0.10006747, 0.19992745], [-0.10031497, -0.19968502, -0.10001184, -0.19998728], [ 0.10065535, 0.19934466, 0.10002466, 0.19997349], [-0.10091198, -0.19908802, -0.10003337, -0.19996412], [ 0.10192172, 0.19807829, 0.10007031, 0.19992441], [-0.05844011, -0.24155989, -0.09840713, -0.20171274], [ 0.19192438, 0.10807563, 0.10348865, 0.19624877]]),
                  np.asarray([[-0.10002081, -0.19997771], [ 0.10004336, 0.19995356], [-0.10000925, -0.19999005], [ 0.10001925, 0.19997929], [-0.10003401, -0.19996329], [ 0.10007132, 0.19992301], [-0.09891845, -0.20116071], [ 0.10234038, 0.19748913]]),
                  np.asarray([ 0.00228109, -0.00483371, 0.00089944, -0.00187375, 0.00228505, -0.00483587, -0.09006158, -0.20659775])),
    }
    self.assert_trained_seq2seq_params(expected, places=5, lr=10)

  # ok in dynet, not in torch:
  # def test_emb_weights_two_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20195894, 0.19691031], [-0.30414006, 0.29530725], [-0.40498427, 0.3935459], [-0.5, 0.5]])
  #   self.assert_trained_emb_params(expected, places=4, epochs=2, lr=100)

  # ok in dynet, not in torch
  # def test_emb_grads_two_epochs(self):
  #   expected = np.asarray(
  #     [[ 0, 0], [-1.43307407e-05, -2.18112727e-05], [-2.92414807e-05, -4.44276811e-05], [-3.09737370e-05, -4.77675057e-05], [ 0,  0]])
  #   self.assert_trained_emb_grads(expected, places=9, lr=10, epochs=2)


class TestManualClassifier(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def run_training(self, num_layers=1, bi_encoder=False, epochs=1, lr=0.1):
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
                                                                     NumpyInitializer(lstm_arr_4_2),   # fwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(lstm_arr_4_2),   # bwd_l0_ih
                                                                     NumpyInitializer(lstm_arr_4_1)])] # bwd_l0_hh
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

  #### OK #########

  # def test_loss_basic(self):
  #   self.assert_loss_value(1.386299, places=5)
  #
  # def test_loss_twolayer(self):
  #   self.assert_loss_value(1.386294, places=5, num_layers=2)
  #
  # def test__loss_bidirectional(self):
  #   self.assert_loss_value(1.386302, places=5, bi_encoder=True)
  #
  # def test_loss_two_epochs(self):
  #   self.assert_loss_value(1.386635, places=5, epochs=2, lr=100)
  #
  # def test_loss_five_epochs(self):
  #   self.assert_loss_value(2.661108, places=2, epochs=5, lr=10)
  #
  # def test_emb_weights_two_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.19894804, 0.20147263], [-0.28823119, 0.32002223], [-0.41040528, 0.3818686], [-0.5, 0.5]])
  #   self.assert_trained_emb_params(expected, places=4, epochs=2, lr=100)
  #
  # def test_emb_weights_five_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20250981, 0.19391325], [-0.29897961, 0.30119216], [-0.40397269, 0.39145479], [-0.5, 0.5]])
  #   self.assert_trained_emb_params(expected, places=3, epochs=5, lr=10)
  #
  # def test_emb_grads(self):
  #   expected = np.asarray(
  #     [[0, 0], [1.2468663e-6, 2.49373261e-6], [-5.26151271e-5, -1.05230254e-4], [5.41623740e-5, 1.08324748e-4], [0, 0]])
  #   self.assert_trained_emb_grads(expected, places=9)
  #
  # def test_emb_grads_two_epochs(self):
  #   expected = np.asarray(
  #     [[ 0, 0], [ 1.23475911e-06, 2.46928539e-06], [-5.26270887e-05, -1.05221523e-04], [ 5.41591871e-05, 1.08285341e-04], [ 0, 0]])
  #   self.assert_trained_emb_grads(expected, places=9, epochs=2)
  #
  # def test_emb_grads_five_epochs(self):
  #   expected = np.asarray(
  #     [[ 0, 0], [ 1.20434561e-06, 2.40851659e-06], [-5.26594959e-05, -1.05188665e-04], [ 5.41539921e-05, 1.08175940e-04], [ 0, 0]])
  #   self.assert_trained_emb_grads(expected, places=8, epochs=5)



if __name__ == '__main__':
  unittest.main()
