"""
Manually initialized toy model, test for both DyNet and Pytorch backends to produce exactly the same loss value.
"""

import unittest

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt import event_trigger
from xnmt.param_initializers import NumpyInitializer, InitializerSequence
from xnmt.modelparts.attenders import DotAttender, MlpAttender
from xnmt.batchers import SrcBatcher
from xnmt.models.classifiers import SequenceClassifier
from xnmt.modelparts.bridges import NoBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import NoopEmbedder, SimpleWordEmbedder
import xnmt.events
from xnmt.input_readers import PlainTextReader, IDReader, H5Reader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculators import MLELoss
from xnmt.loss_trackers import TrainLossTracker
from xnmt.transducers.pyramidal import PyramidalLSTMSeqTransducer
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

  def assert_trained_src_emb_params(self, val, places, *args, **kwargs):
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

  def assert_trained_mlp_att_grads(self, val, places, epochs=1, *args, **kwargs):
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
      actual_grads = training_regimen.model.attender.pW.grad_as_array(), \
                     training_regimen.model.attender.pV.grad_as_array(), \
                     training_regimen.model.attender.pb.grad_as_array(), \
                     training_regimen.model.attender.pU.grad_as_array()
    else:
      actual_grads = tt.npvalue(training_regimen.model.attender.linear_context._parameters['weight'].grad).T, \
                     tt.npvalue(training_regimen.model.attender.linear_query._parameters['weight'].grad).T, \
                     tt.npvalue(training_regimen.model.attender.linear_query._parameters['bias'].grad).T, \
                     tt.npvalue(training_regimen.model.attender.pU._parameters['weight'].grad).T,
    np.testing.assert_almost_equal(actual_grads[0], val[0], decimal=places)
    np.testing.assert_almost_equal(actual_grads[1], val[1], decimal=places)
    np.testing.assert_almost_equal(actual_grads[2], val[2], decimal=places)
    np.testing.assert_almost_equal(actual_grads[3].flatten(), val[3].flatten(), decimal=places)

  def convert_pytorch_lstm_weights(self, weights):
    # change ifgo -> ifog; subtract 1-initialized forget gates
    h_dim = weights[0].shape[1] // 4
    return np.concatenate([weights[0][:, :h_dim * 2], weights[0][:, h_dim * 3:],
                           weights[0][:, h_dim * 2:h_dim * 3]], axis=1), \
           np.concatenate([weights[1][:, :h_dim * 2], weights[1][:, h_dim * 3:],
                           weights[1][:, h_dim * 2:h_dim * 3]], axis=1), \
           np.concatenate([weights[2][:h_dim], weights[2][h_dim:h_dim * 2] - 1,
                           weights[2][h_dim * 3:], weights[2][h_dim * 2:h_dim * 3]], axis=0),

class TestManualFullLAS(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def assert_trained_seq2seq_params(self, val, places, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    if xnmt.backend_dynet:
      trained_enc_l0_fwd = training_regimen.model.encoder.builder_layers[0][0].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[0][0].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[0][0].b[0].as_array()
      trained_enc_l0_bwd = training_regimen.model.encoder.builder_layers[0][1].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[0][1].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[0][1].b[0].as_array()
      trained_enc_l1_fwd = training_regimen.model.encoder.builder_layers[1][0].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[1][0].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[1][0].b[0].as_array()
      trained_enc_l1_bwd = training_regimen.model.encoder.builder_layers[1][1].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[1][1].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[1][1].b[0].as_array()
      trained_enc_l2_fwd = training_regimen.model.encoder.builder_layers[2][0].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[2][0].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[2][0].b[0].as_array()
      trained_enc_l2_bwd = training_regimen.model.encoder.builder_layers[2][1].Wx[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[2][1].Wh[0].as_array(), \
                           training_regimen.model.encoder.builder_layers[2][1].b[0].as_array()
      trained_decoder = training_regimen.model.decoder.rnn.Wx[0].as_array(), \
                        training_regimen.model.decoder.rnn.Wh[0].as_array(), \
                        training_regimen.model.decoder.rnn.b[0].as_array()
      trained_trg_emb = training_regimen.model.decoder.embedder.embeddings.as_array()
      trained_transform = training_regimen.model.decoder.transform.W1.as_array(), \
                          training_regimen.model.decoder.transform.b1.as_array()
      trained_out = training_regimen.model.decoder.scorer.output_projector.W1.as_array(), \
                    training_regimen.model.decoder.scorer.output_projector.b1.as_array()
      trained_attender = training_regimen.model.attender.pV.as_array(), \
                         training_regimen.model.attender.pW.as_array(), \
                         training_regimen.model.attender.pb.as_array(), \
                         training_regimen.model.attender.pU.as_array()
    else:
      trained_trg_emb = tt.npvalue(training_regimen.model.decoder.embedder.embeddings._parameters['weight'].data)
      trained_transform = tt.npvalue(training_regimen.model.decoder.transform.linear._parameters['weight'].data), \
                          tt.npvalue(training_regimen.model.decoder.transform.linear._parameters['bias'].data)
      trained_out= tt.npvalue(training_regimen.model.decoder.scorer.output_projector.linear._parameters['weight'].data), \
                   tt.npvalue(training_regimen.model.decoder.scorer.output_projector.linear._parameters['bias'].data)
      trained_enc_l0_fwd = tt.npvalue(training_regimen.model.encoder.builder_layers[0][0].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[0][0].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[0][0].layers[0]._parameters['bias_ih'].data)
      trained_enc_l0_bwd = tt.npvalue(training_regimen.model.encoder.builder_layers[0][1].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[0][1].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[0][1].layers[0]._parameters['bias_ih'].data)
      trained_enc_l1_fwd = tt.npvalue(training_regimen.model.encoder.builder_layers[1][0].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[1][0].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[1][0].layers[0]._parameters['bias_ih'].data)
      trained_enc_l1_bwd = tt.npvalue(training_regimen.model.encoder.builder_layers[1][1].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[1][1].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[1][1].layers[0]._parameters['bias_ih'].data)
      trained_enc_l2_fwd = tt.npvalue(training_regimen.model.encoder.builder_layers[2][0].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[2][0].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[2][0].layers[0]._parameters['bias_ih'].data)
      trained_enc_l2_bwd = tt.npvalue(training_regimen.model.encoder.builder_layers[2][1].layers[0]._parameters['weight_ih'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[2][1].layers[0]._parameters['weight_hh'].data), \
                           tt.npvalue(training_regimen.model.encoder.builder_layers[2][1].layers[0]._parameters['bias_ih'].data)
      trained_enc_l0_fwd = self.convert_pytorch_lstm_weights(trained_enc_l0_fwd)
      trained_enc_l0_bwd = self.convert_pytorch_lstm_weights(trained_enc_l0_bwd)
      trained_enc_l1_fwd = self.convert_pytorch_lstm_weights(trained_enc_l1_fwd)
      trained_enc_l1_bwd = self.convert_pytorch_lstm_weights(trained_enc_l1_bwd)
      trained_enc_l2_fwd = self.convert_pytorch_lstm_weights(trained_enc_l2_fwd)
      trained_enc_l2_bwd = self.convert_pytorch_lstm_weights(trained_enc_l2_bwd)
      trained_decoder = tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_ih'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_hh'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['bias_ih'].data)
      trained_attender = tt.npvalue(training_regimen.model.attender.linear_context._parameters['weight'].data), \
                         tt.npvalue(training_regimen.model.attender.linear_query._parameters['weight'].data), \
                         tt.npvalue(training_regimen.model.attender.linear_query._parameters['bias'].data), \
                         tt.npvalue(training_regimen.model.attender.pU._parameters['weight'].data)
      trained_decoder = self.convert_pytorch_lstm_weights(trained_decoder)
      for k,v in val.items():
        if v is None: continue
        if type(v)==tuple: val[k] = tuple(vi.T for vi in v)
        else: val[k] = v.T
    if val['trg_emb'] is not None:
      np.testing.assert_almost_equal(trained_trg_emb, val['trg_emb'], decimal=places)
    if val['transform'] is not None:
      np.testing.assert_almost_equal(trained_transform[0], val['transform'][0], decimal=places)
      np.testing.assert_almost_equal(trained_transform[1], val['transform'][1], decimal=places)
    if val['out'] is not None:
      np.testing.assert_almost_equal(trained_out[0], val['out'][0], decimal=places)
      np.testing.assert_almost_equal(trained_out[1], val['out'][1], decimal=places)
    if val['enc_l0_fwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l0_fwd[0], val['enc_l0_fwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l0_fwd[1], val['enc_l0_fwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l0_fwd[2], val['enc_l0_fwd'][2], decimal=places)
    if val['enc_l0_bwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l0_bwd[0], val['enc_l0_bwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l0_bwd[1], val['enc_l0_bwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l0_bwd[2], val['enc_l0_bwd'][2], decimal=places)
    if val['enc_l1_fwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l1_fwd[0], val['enc_l1_fwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l1_fwd[1], val['enc_l1_fwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l1_fwd[2], val['enc_l1_fwd'][2], decimal=places)
    if val['enc_l1_bwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l1_bwd[0], val['enc_l1_bwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l1_bwd[1], val['enc_l1_bwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l1_bwd[2], val['enc_l1_bwd'][2], decimal=places)
    if val['enc_l2_fwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l2_fwd[0], val['enc_l2_fwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l2_fwd[1], val['enc_l2_fwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l2_fwd[2], val['enc_l2_fwd'][2], decimal=places)
    if val['enc_l2_bwd'] is not None:
      np.testing.assert_almost_equal(trained_enc_l2_bwd[0], val['enc_l2_bwd'][0], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l2_bwd[1], val['enc_l2_bwd'][1], decimal=places)
      np.testing.assert_almost_equal(trained_enc_l2_bwd[2], val['enc_l2_bwd'][2], decimal=places)
    if val['decoder'] is not None:
      np.testing.assert_almost_equal(trained_decoder[0], val['decoder'][0], decimal=places)
      np.testing.assert_almost_equal(trained_decoder[1], val['decoder'][1], decimal=places)
      np.testing.assert_almost_equal(trained_decoder[2], val['decoder'][2], decimal=places)
    if val['attender'] is not None:
      # below 3 are mismatched:
      np.testing.assert_almost_equal(trained_attender[0], val['attender'][0], decimal=places)
      np.testing.assert_almost_equal(trained_attender[1], val['attender'][1], decimal=places)
      np.testing.assert_almost_equal(trained_attender[2], val['attender'][2], decimal=places)
      np.testing.assert_almost_equal(trained_attender[3].flatten(), val['attender'][3].flatten(), decimal=places)

  def run_training(self, epochs=1, lr=0.1, adam=False):
    # TODO: AdamTrainer, lr_decay, restart_trainer
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False, pad_src_to_multiple=4)
    train_args = {}
    train_args['src_file'] = "test/data/LDC94S13A.h5"
    train_args['trg_file'] = "test/data/ab-ba-ab-ba-ab.txt"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    emb_arr_5_2 = np.asarray([[-0.1, 0.1],[-0.2, 0.2],[-0.3, 0.3],[-0.4, 0.4],[-0.5, 0.5],])
    proj_arr_2_4 = np.asarray([
      [-0.1, -0.2, -0.3, -0.4],
      [0.1, 0.2, 0.3, 0.4],
    ])
    proj_arr_2_2 = np.asarray([
      [-0.1, -0.2],
      [0.1, 0.2],
    ])
    proj_arr_1_2 = np.asarray([
      [-0.1, -0.2],
    ])
    # note: dynet uses i|f|o|g, while pytorch uses i|f|g|o order; must make sure to initialize output and update matrices to the same value
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
    encoder = PyramidalLSTMSeqTransducer(input_dim=layer_dim,
                                         hidden_dim=layer_dim,
                                         downsampling_method='skip',
                                         param_init=InitializerSequence([
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # fwd_l0_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # fwd_l0_hh
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # bwd_l0_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # bwd_l0_hh
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # bwd_l1_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # bwd_l1_hh
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # bwd_l1_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # bwd_l1_hh
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # bwd_l2_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # bwd_l2_hh
                                           InitializerSequence([
                                             NumpyInitializer(lstm_arr_4_2),  # bwd_l2_ih
                                             NumpyInitializer(lstm_arr_4_1)]),  # bwd_l2_hh
                                         ]),
                                         layers=3)
    train_args['model'] = \
      DefaultTranslator(
        src_reader=H5Reader(transpose=True, feat_to=2),
        src_embedder=NoopEmbedder(emb_dim=layer_dim),
        encoder=encoder,
        attender=MlpAttender(input_dim=layer_dim, hidden_dim=layer_dim, state_dim=2, param_init=InitializerSequence([NumpyInitializer(proj_arr_2_2), NumpyInitializer(proj_arr_2_2), NumpyInitializer(proj_arr_1_2)])),
        decoder=AutoRegressiveDecoder(
          input_dim=layer_dim,
          embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(emb_arr_5_2),
                                      fix_norm=1),
          rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                   hidden_dim=layer_dim,
                                   decoder_input_dim=layer_dim,
                                   layers=1,
                                   param_init=InitializerSequence(
                                     [NumpyInitializer(dec_lstm_arr_8_4)] + [NumpyInitializer(lstm_arr_8_2)]),
                                   yaml_path="model.decoder.rnn"),
          transform=NonLinear(input_dim=layer_dim * 2, output_dim=layer_dim, param_init=NumpyInitializer(proj_arr_2_4)),
          scorer=Softmax(input_dim=layer_dim, vocab_size=vocab_size, label_smoothing=0.1,
                         param_init=NumpyInitializer(emb_arr_5_2)),
          bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
        trg_reader=PlainTextReader(vocab=vocab),
      )
    train_args['dev_tasks'] = []
    if xnmt.backend_dynet:
      if adam:
        train_args['trainer'] = optimizers.AdamTrainer(alpha=lr*0.01, clip_grads=-1)
      else:
        train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=-1)
    else:
      if adam:
        train_args['trainer'] = optimizers.AdamTrainer(alpha=lr * 0.01, clip_grads=0, rescale_grads=False)
      else:
        train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=0, rescale_grads=False)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = epochs
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    train_args['max_num_train_sents'] = 2
    train_args['loss_comb_method'] = 'avg'
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  # def test_loss_basic(self):
  #   self.assert_loss_value(9.657503128051758, places=5)
  #
  # def test_loss_three_epochs(self):
  #   self.assert_loss_value(8.524262428283691, places=2, epochs=3, lr=10)
  #
  # def test_mlp_att_grads(self):
  #   expected = (np.asarray([[-3.15510178e-08,-3.91636625e-08],[-6.31020356e-08,-7.83273251e-08]]),
  #               np.asarray([[-1.15501116e-10,1.41344131e-10],[-2.31002231e-10,2.82688262e-10]]),
  #               np.asarray([1.65015179e-09,3.30030359e-09]),
  #               np.asarray([[-1.09920165e-07,1.09920165e-07]]))
  #   self.assert_trained_mlp_att_grads(expected, places=7, epochs=1, adam=True)

  # def test_mlp_att_grads_two_epochs_adam(self):
  #   expected = (np.asarray([[1.9886029e-05, 2.2218173e-05], [-5.5424091e-05, -6.2103529e-05]]),
  #               np.asarray([[-1.9301253e-07, -1.1546918e-08], [7.1632535e-06, 4.7493609e-07]]),
  #               np.asarray([ 3.0994852e-07, -1.1994176e-05]),
  #               np.asarray([-1.3820918e-05,  8.8202738e-05]))
  #   self.assert_trained_mlp_att_grads(expected, places=7, epochs=2, adam=True, lr=100)

  # def test_mlp_att_grads_trained(self):
  #   expected = (np.asarray([[ 6.15145327e-05,5.41474146e-05],[-1.41641664e-04,-1.24026701e-04]]),
  #               np.asarray([[-9.45401825e-06,-1.82743112e-04],[ 1.28936317e-05,2.54007202e-04]]),
  #               np.asarray([-0.00021929,0.0002702,]),
  #               np.asarray([[ 0.00020377,-0.00013509]]))
  #   self.assert_trained_mlp_att_grads(expected, places=7, epochs=3, adam=True, lr=100)

  # def test_loss_three_epochs_adam(self):
  #   self.assert_loss_value(8.912698745727539, places=2, epochs=3, lr=10, adam=True)
  # def test_loss_ten_epochs_adam(self):
  #   self.assert_loss_value(7.670589447021484, places=2, epochs=10, lr=10, adam=True)

  # def test_all_params_one_epoch(self):
  #   # useful regex: (?<!\[)\s+      ->      ,
  #   expected = {
  #     'trg_emb': np.asarray([[-0.08201125,0.11798876],[-0.2,0.2,],[-0.30986509,0.29013494],[-0.41663632,0.38336369],[-0.5,0.5,]]),
  #     'transform': (np.asarray([[-0.11458226,-0.1818178,-0.29989332,-0.39990845],[ 0.11458226,0.1818178,0.29989332,0.39990845]]),
  #                   np.asarray([ 0.01091676,-0.01091676])),
  #     'out': (np.asarray([[-0.05089095,0.05089095],[-0.2470859,0.2470859,],[-0.32606718,0.32606718],[-0.42551777,0.42551777],[-0.45043823,0.45043823]]),
  #             np.asarray([-5.37809563,3.61097455,3.60002279,3.58904791,-5.42194843])),
  #     'enc_l0_fwd': (np.asarray([[-0.09993633,-0.19993669],[-0.09990665,-0.19991051],[-0.09992233,-0.19992436],[-0.09978127,-0.1998006,]]),
  #                    np.asarray([[-0.10001528],[-0.10002846],[-0.1000223,],[-0.10008691]]),
  #                    np.asarray([-1.90493865e-05,-5.03285955e-05,-3.61094171e-05,-4.47910599e-04])),
  #     'enc_l0_bwd': (np.asarray([[-0.09987763,-0.19987726],[-0.09983368,-0.19983764],[-0.09984706,-0.1998519,],[-0.09962509,-0.19965594]]),
  #                    np.asarray([[-0.10002756],[-0.10004743],[-0.100041,],[-0.10013461]]),
  #                    np.asarray([-3.16269543e-05,-6.89425797e-05,-6.18750637e-05,-8.94205179e-04])),
  #     'enc_l1_fwd': (np.asarray([[-0.10003329,-0.20003363],[-0.10004379,-0.20003976],[-0.10003274,-0.20003073],[-0.09949255,-0.19953378]]),
  #                    np.asarray([[-0.09999352],[-0.09999096],[-0.09999333],[-0.10010429]]),
  #                    np.asarray([-7.18495066e-05,-9.96304589e-05,-7.29403910e-05,2.58315331e-03])),
  #     'enc_l1_bwd': (np.asarray([[-0.10006139,-0.20005998],[-0.10006943,-0.2000725,],[-0.10005932,-0.20006149],[-0.099105,-0.19919929]]),
  #                    np.asarray([[-0.09998865],[-0.09998529],[-0.09998781],[-0.10014705]]),
  #                    np.asarray([-0.00012661,-0.00014598,-0.0001292,0.00507051])),
  #     'enc_l2_fwd': (np.asarray([[-0.09999367,-0.19999345],[-0.09999148,-0.1999929,],[-0.09999393,-0.19999467],[-0.09944947,-0.19950114]]),
  #                    np.asarray([[-0.10000134],[-0.10000204],[-0.10000141],[-0.10013182]]),
  #                    np.asarray([-6.99323864e-05,-1.00616948e-04,-7.01395402e-05,-1.25058405e-02])),
  #     'enc_l2_bwd': (np.asarray([[-0.09998975,-0.19999012],[-0.09998945,-0.19998839],[-0.09999086,-0.1999899,],[-0.09914374,-0.19928391]]),
  #                    np.asarray([[-0.10000202],[-0.10000266],[-0.10000228],[-0.10015319]]),
  #                    np.asarray([-0.00010539,-0.00011597,-0.00010555,-0.02152999])),
  #     'decoder': (np.asarray([[-0.10115287,-0.19884713,-0.0999788,-0.19998185],[ 0.10264169,0.19735833,0.09995335,0.19996007],[-0.10086214,-0.19913787,-0.09998852,-0.19999012],[ 0.10191832,0.19808169,0.09997454,0.19997808],[-0.10116317,-0.19883683,-0.09997369,-0.19997761],[ 0.10266069,0.19733933,0.0999413,0.19995002],[-0.07204451,-0.22795549,-0.10052628,-0.20045052],[ 0.16965774,0.13034226,0.09874846,0.19892833]]),
  #                 np.asarray([[-0.10005526,-0.19993505],[ 0.10012097,0.19985785],[-0.10003345,-0.19996063],[ 0.10007406,0.19991286],[-0.10008492,-0.19989935],[ 0.1001896,0.19977526],[-0.0986231,-0.20161855],[ 0.10325646,0.19617321]]),
  #                 np.asarray([ 0.0016304,-0.00373591,0.00121925,-0.00271291,0.00164497,-0.00376278,-0.03953505,-0.09851094])),
  #     'attender': (np.asarray([[-0.1,-0.2],[ 0.1,0.2]]), # TODO: why are these untrained?
  #                   np.asarray([[-0.09999969,-0.19999962],[ 0.10000063,0.20000079]]),
  #                   np.asarray([-1.65015184e-08,-3.30030367e-08]),
  #                   np.asarray([[-0.0999989,-0.20000111]])),
  #   }
  #   self.assert_trained_seq2seq_params(expected, places=6, lr=10)

  def test_all_params_one_epoch_sgd_must_fail(self):
    # useful regex: (?<!\[)\s+      ->      ,
    expected = {
      'trg_emb': None,
      'transform': None,
      'out': None,
      'enc_l0_fwd': None,
      'enc_l0_bwd': None,
      'enc_l1_fwd': None,
      'enc_l1_bwd': None,
      'enc_l2_fwd': None,
      'enc_l2_bwd': None,
      'decoder': None,
      'attender': (np.asarray([[-0.100004, -0.200005], [ 0.099993,  0.199991]]),
                    np.asarray([[-0.099909, -0.199649], [ 0.100183,  0.200702]]),
                    np.asarray([1.710342e-05, 3.421417e-05]),
                    np.asarray([-0.0992, -0.2008])),
    }
    self.assert_trained_seq2seq_params(expected, places=6, lr=100, epochs=10)

  # def test_all_params_one_epoch_sgd_must_not_fail(self):
  #   # useful regex: (?<!\[)\s+      ->      ,
  #   expected = {
  #     'trg_emb': np.asarray([[-0.46045405 ,0.43411583] ,[-0.2 ,0.2 ,] ,[-0.39914325 ,0.20037407] ,[-0.54675782 ,0.28116238] ,[-0.5 ,0.5 ,]]),
  #     'transform': None,
  #     'out': None,
  #     'enc_l0_fwd': None,
  #     'enc_l0_bwd': None,
  #     'enc_l1_fwd': None,
  #     'enc_l1_bwd': None,
  #     'enc_l2_fwd': None,
  #     'enc_l2_bwd': None,
  #     'decoder': None,
  #     'attender':None,
  #   }
  #   self.assert_trained_seq2seq_params(expected, places=5, lr=100, epochs=10)


  # def test_all_params_one_epoch_adam(self):
  #   # useful regex: (?<!\[)\s+      ->      ,
  #   expected = {
  #     'trg_emb': np.asarray([[-1.75759196e-05,1.99982435e-01],[-2.00000003e-01,2.00000003e-01],[-3.99967968e-01,2.00032040e-01],[-4.99980986e-01,3.00019026e-01],[-5.00000000e-01,5.00000000e-01]]),
  #     'transform': (np.asarray([[-0.19997832,-0.1000174,-0.20287874,-0.3033388,],[ 0.19997832,0.1000174,0.20287874,0.3033388,]]),
  #                   np.asarray([ 0.09997104,-0.09997104])),
  #     'out': (np.asarray([[-6.44475222e-06,6.44475222e-06],[-2.99993277e-01,2.99993277e-01],[-3.99987876e-01,3.99987876e-01],[-4.99987602e-01,4.99987602e-01],[-4.00006384e-01,4.00006384e-01]]),
  #             np.asarray([-0.09999994,0.09999993,0.09999992,0.0999999,-0.09999994])),
  #     'enc_l0_fwd': (np.asarray([[-0.00473128,-0.1047574,],[-0.00327653,-0.10341273],[-0.00391212,-0.10401335],[-0.00142514,-0.10156112]]),
  #                    np.asarray([[-0.1828548,],[-0.18999776],[-0.18757865],[-0.19648919]]),
  #                    np.asarray([-0.08576292,-0.09408816,-0.09194765,-0.09929894])),
  #     'enc_l0_bwd': (np.asarray([[-0.00251901,-0.10251159],[-0.00186585,-0.10191042],[-0.00202568,-0.10209057],[-0.00083642,-0.10091073]]),
  #                    np.asarray([[-0.18970746],[-0.19374949],[-0.1928391,],[-0.1977047,]]),
  #                    np.asarray([-0.09091013,-0.0956143,-0.09513772,-0.09964762])),
  #     'enc_l1_fwd': (np.asarray([[-0.19132513,-0.29140514],[-0.19326439,-0.29263091],[-0.19119145,-0.29066783],[-0.00061932,-0.1006737,]]),
  #                    np.asarray([[-0.03277878],[-0.02590235],[-0.03216105],[-0.19705713]]),
  #                    np.asarray([-0.09578426,-0.09692363,-0.0958447,0.09987774])),
  #     'enc_l1_bwd': (np.asarray([[-0.19510096,-0.2949917,],[-0.19564378,-0.29581988],[-0.19493818,-0.29510871],[-0.00035208,-0.10039338]]),
  #                    np.asarray([[-0.02177877],[-0.01768639],[-0.02059451],[-0.19789474]]),
  #                    np.asarray([-0.09756327,-0.09787969,-0.09761088,0.09993766])),
  #     'enc_l2_fwd': (np.asarray([[-0.03331528,-0.13254735],[-0.02706053,-0.13079686],[-0.03424625,-0.13719964],[-0.00057114,-0.10062992]]),
  #                    np.asarray([[-0.12979138],[-0.13919146],[-0.13082246],[-0.19765723]]),
  #                    np.asarray([-0.09567371,-0.09695286,-0.09568593,-0.09997474])),
  #     'enc_l2_bwd': (np.asarray([[-0.02356844,-0.12424017],[-0.02306087,-0.12141666],[-0.0256956,-0.12383554],[-0.00036795,-0.10043966]]),
  #                    np.asarray([[-0.13893157],[-0.14568059],[-0.14186357],[-0.19797748]]),
  #                    np.asarray([-0.09708674,-0.09734552,-0.09709109,-0.09998534])),
  #     'decoder': (np.asarray([[-1.99726462e-01,-1.00273542e-01,-1.29794180e-02,-1.14841036e-01],[ 1.99880436e-01,1.00119568e-01,6.34786487e-03,1.07336700e-01],[-1.99634552e-01,-1.00365460e-01,-2.15895846e-02,-1.24240689e-01],[ 1.99835435e-01,1.00164570e-01,1.10447034e-02,1.12606354e-01],[-1.99728876e-01,-1.00271128e-01,-1.07286796e-02,-1.12374164e-01],[ 1.99881285e-01,1.00118719e-01,5.11157513e-03,1.05950944e-01],[-1.13099813e-05,-2.99988687e-01,-1.99402705e-01,-2.99302965e-01],[ 1.99995473e-01,1.00004531e-01,2.52030790e-04,1.00294210e-01]]),
  #                 np.asarray([[-1.94587052e-01,-1.04642287e-01],[ 1.97452396e-01,1.02176145e-01],[-1.91362619e-01,-1.07435435e-01],[ 1.95904762e-01,1.03501700e-01],[-1.96409553e-01,-1.03045903e-01],[ 1.98359489e-01,1.01387605e-01],[-2.29150057e-04,-2.99805015e-01],[ 1.99902982e-01,1.00082561e-01]]),
  #                 np.asarray([ 0.09980642,-0.09991542,0.09974132,-0.09988359,0.09980813,-0.09991604,-0.09999201,-0.09999679])),
  #     'attender': (np.asarray([[-0.09996349,-0.20004468],[ 0.10007299,0.19991069]]),
  #                   np.asarray([[-0.09092791,-0.18898019],[ 0.11663505,0.21985196]]),
  #                   np.asarray([-0.00051911,-0.00103286]),
  #                   np.asarray([[-0.07420623,-0.22579378]])),
  #   }
  #   self.assert_trained_seq2seq_params(expected, places=6, lr=10, adam=True)


  # def test_emb_weights_one_epoch(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20184304, 0.19631392], [-0.30349943, 0.29300117], [-0.40391687, 0.39216626], [-0.5, 0.5]])
  #   self.assert_trained_src_emb_params(expected, places=5, lr=100)
  #
  # def test_emb_grads_one_epoch(self):
  #   expected = np.asarray(
  #     [[0, 0], [1.84304245e-5, 3.68608489e-5], [3.49941438e-5, 6.99882876e-5], [3.91686735e-5, 7.83373471e-5], [0, 0]])
  #   self.assert_trained_emb_grads(expected, places=9, lr=10)
  #
  #
  # def test_all_grads_one_epoch(self):
  #   expected = {
  #     'src_emb': np.asarray([[-0.1, 0.1 ], [-0.2001843, 0.19963139], [-0.30034995, 0.29930013], [-0.4003917, 0.39921662], [-0.5, 0.5]]),
  #     'trg_emb': np.asarray([[-0.09716769, 0.10566463], [-0.2, 0.2], [-0.30496135, 0.29007733], [-0.41122547, 0.37754905], [-0.5, 0.5 ]]),
  #     'transform': (np.asarray([[-0.12163746, -0.17607024, -0.30009004, -0.39990318], [ 0.12163746, 0.17607024, 0.30009004, 0.39990318]]),
  #                   np.asarray([ 0.00683933, -0.00683933])),
  #     'out': (np.asarray([[-0.06577028, 0.06577028], [-0.23925397, 0.23925397], [-0.31620809, 0.31620809], [-0.41308054, 0.41308054], [-0.46568716, 0.46568716]]),
  #             np.asarray([-11.9862957, 8.00685501, 8.00000381, 7.99314737, -12.01371288])),
  #     'encoder': (np.asarray([[-0.10004691, -0.19995309], [ 0.10009123, 0.19990878], [-0.10001086, -0.19998914], [ 0.10002112, 0.19997889], [-0.1000402, -0.19995981], [ 0.10007835, 0.19992165], [-0.09733033, -0.20266968], [ 0.10536621,  0.1946338 ]]),
  #                 np.asarray([[-0.1000007, -0.19999924], [ 0.10000135, 0.19999854], [-0.10000048, -0.19999948], [ 0.10000093, 0.19999899], [-0.10000127, -0.19999863], [ 0.10000247, 0.19999734], [-0.09995046, -0.20005347], [ 0.10009786, 0.19989437]]),
  #                 np.asarray([ 1.38808842e-04, -2.69481330e-04, 4.07925472e-05, -7.95210435e-05, 1.38590593e-04, -2.70244025e-04, -8.30172468e-03, -1.66236106e-02])),
  #     'decoder': (np.asarray([[-0.10089689, -0.19910312, -0.10003209, -0.19996549], [ 0.10188823, 0.19811177, 0.10006747, 0.19992745], [-0.10031497, -0.19968502, -0.10001184, -0.19998728], [ 0.10065535, 0.19934466, 0.10002466, 0.19997349], [-0.10091198, -0.19908802, -0.10003337, -0.19996412], [ 0.10192172, 0.19807829, 0.10007031, 0.19992441], [-0.05844011, -0.24155989, -0.09840713, -0.20171274], [ 0.19192438, 0.10807563, 0.10348865, 0.19624877]]),
  #                 np.asarray([[-0.10002081, -0.19997771], [ 0.10004336, 0.19995356], [-0.10000925, -0.19999005], [ 0.10001925, 0.19997929], [-0.10003401, -0.19996329], [ 0.10007132, 0.19992301], [-0.09891845, -0.20116071], [ 0.10234038, 0.19748913]]),
  #                 np.asarray([ 0.00228109, -0.00483371, 0.00089944, -0.00187375, 0.00228505, -0.00483587, -0.09006158, -0.20659775])),
  #   }
  #   self.assert_trained_seq2seq_params(expected, places=5, lr=10)
  #
  # # ok in dynet, not in torch:
  # def test_emb_weights_two_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20195894, 0.19691031], [-0.30414006, 0.29530725], [-0.40498427, 0.3935459], [-0.5, 0.5]])
  #   self.assert_trained_src_emb_params(expected, places=4, epochs=2, lr=100)
  #
  # # ok in dynet, not in torch
  # def test_emb_grads_two_epochs(self):
  #   expected = np.asarray(
  #     [[ 0, 0], [-1.43307407e-05, -2.18112727e-05], [-2.92414807e-05, -4.44276811e-05], [-3.09737370e-05, -4.77675057e-05], [ 0,  0]])
  #   self.assert_trained_emb_grads(expected, places=8, lr=10, epochs=2)

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
      trained_encoder = self.convert_pytorch_lstm_weights(trained_encoder)
      trained_decoder = tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_ih'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['weight_hh'].data), \
                        tt.npvalue(training_regimen.model.decoder.rnn.layers[0]._parameters['bias_ih'].data)
      trained_decoder = self.convert_pytorch_lstm_weights(trained_decoder)
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
    # note: dynet uses i|f|o|g, while pytorch uses i|f|g|o order; must make sure to initialize output and update matrices to the same value
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
  #   self.assert_trained_src_emb_params(expected, places=5, lr=100)
  #
  # def test_emb_grads_one_epoch(self):
  #   expected = np.asarray(
  #     [[0, 0], [1.84304245e-5, 3.68608489e-5], [3.49941438e-5, 6.99882876e-5], [3.91686735e-5, 7.83373471e-5], [0, 0]])
  #   self.assert_trained_emb_grads(expected, places=9, lr=10)
  #
  #
  # def test_all_params_one_epoch(self):
  #   expected = {
  #     'src_emb': np.asarray([[-0.1, 0.1 ], [-0.2001843, 0.19963139], [-0.30034995, 0.29930013], [-0.4003917, 0.39921662], [-0.5, 0.5]]),
  #     'trg_emb': np.asarray([[-0.09716769, 0.10566463], [-0.2, 0.2], [-0.30496135, 0.29007733], [-0.41122547, 0.37754905], [-0.5, 0.5 ]]),
  #     'transform': (np.asarray([[-0.12163746, -0.17607024, -0.30009004, -0.39990318], [ 0.12163746, 0.17607024, 0.30009004, 0.39990318]]),
  #                   np.asarray([ 0.00683933, -0.00683933])),
  #     'out': (np.asarray([[-0.06577028, 0.06577028], [-0.23925397, 0.23925397], [-0.31620809, 0.31620809], [-0.41308054, 0.41308054], [-0.46568716, 0.46568716]]),
  #             np.asarray([-11.9862957, 8.00685501, 8.00000381, 7.99314737, -12.01371288])),
  #     'encoder': (np.asarray([[-0.10004691, -0.19995309], [ 0.10009123, 0.19990878], [-0.10001086, -0.19998914], [ 0.10002112, 0.19997889], [-0.1000402, -0.19995981], [ 0.10007835, 0.19992165], [-0.09733033, -0.20266968], [ 0.10536621,  0.1946338 ]]),
  #                 np.asarray([[-0.1000007, -0.19999924], [ 0.10000135, 0.19999854], [-0.10000048, -0.19999948], [ 0.10000093, 0.19999899], [-0.10000127, -0.19999863], [ 0.10000247, 0.19999734], [-0.09995046, -0.20005347], [ 0.10009786, 0.19989437]]),
  #                 np.asarray([ 1.38808842e-04, -2.69481330e-04, 4.07925472e-05, -7.95210435e-05, 1.38590593e-04, -2.70244025e-04, -8.30172468e-03, -1.66236106e-02])),
  #     'decoder': (np.asarray([[-0.10089689, -0.19910312, -0.10003209, -0.19996549], [ 0.10188823, 0.19811177, 0.10006747, 0.19992745], [-0.10031497, -0.19968502, -0.10001184, -0.19998728], [ 0.10065535, 0.19934466, 0.10002466, 0.19997349], [-0.10091198, -0.19908802, -0.10003337, -0.19996412], [ 0.10192172, 0.19807829, 0.10007031, 0.19992441], [-0.05844011, -0.24155989, -0.09840713, -0.20171274], [ 0.19192438, 0.10807563, 0.10348865, 0.19624877]]),
  #                 np.asarray([[-0.10002081, -0.19997771], [ 0.10004336, 0.19995356], [-0.10000925, -0.19999005], [ 0.10001925, 0.19997929], [-0.10003401, -0.19996329], [ 0.10007132, 0.19992301], [-0.09891845, -0.20116071], [ 0.10234038, 0.19748913]]),
  #                 np.asarray([ 0.00228109, -0.00483371, 0.00089944, -0.00187375, 0.00228505, -0.00483587, -0.09006158, -0.20659775])),
  #   }
  #   self.assert_trained_seq2seq_params(expected, places=5, lr=10)
  #
  # # ok in dynet, not in torch:
  # def test_emb_weights_two_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20195894, 0.19691031], [-0.30414006, 0.29530725], [-0.40498427, 0.3935459], [-0.5, 0.5]])
  #   self.assert_trained_src_emb_params(expected, places=4, epochs=2, lr=100)
  #
  # # ok in dynet, not in torch
  # def test_emb_grads_two_epochs(self):
  #   expected = np.asarray(
  #     [[ 0, 0], [-1.43307407e-05, -2.18112727e-05], [-2.92414807e-05, -4.44276811e-05], [-3.09737370e-05, -4.77675057e-05], [ 0,  0]])
  #   self.assert_trained_emb_grads(expected, places=8, lr=10, epochs=2)


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
  #   self.assert_trained_src_emb_params(expected, places=4, epochs=2, lr=100)
  #
  # def test_emb_weights_five_epochs(self):
  #   expected = np.asarray(
  #     [[-0.1, 0.1], [-0.20250981, 0.19391325], [-0.29897961, 0.30119216], [-0.40397269, 0.39145479], [-0.5, 0.5]])
  #   self.assert_trained_src_emb_params(expected, places=3, epochs=5, lr=10)
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
