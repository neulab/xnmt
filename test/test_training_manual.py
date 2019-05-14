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

TEST_ALL = True

np.set_printoptions(floatmode='unique', linewidth=10000, edgeitems=100)

class ManualTestingBaseClass(object):

  EMB_ARR_5_2 = np.asarray([[-0.1, 0.1], [-0.2, 0.2], [-0.3, 0.3], [-0.4, 0.4], [-0.5, 0.5], ])
  OUT_ARR_2_2 = np.asarray([[-0.1, 0.1], [-0.2, 0.2], ])
  PROJ_ARR_1_2 = np.asarray([
    [-0.1, -0.2],
  ])
  PROJ_ARR_2_2 = np.asarray([
    [-0.1, -0.2],
    [0.1, 0.2],
  ])
  PROJ_ARR_2_4 = np.asarray([
    [-0.1, -0.2, -0.3, -0.4],
    [0.1, 0.2, 0.3, 0.4],
  ])
  # note: dynet uses i|f|o|g, while pytorch uses i|f|g|o order; must make sure to initialize output and update matrices to the same value
  LSTM_ARR_8_2 = np.asarray([
    [-0.1, -0.2],
    [0.1, 0.2],
    [-0.1, -0.2],
    [0.1, 0.2],
    [-0.1, -0.2],
    [0.1, 0.2],
    [-0.1, -0.2],
    [0.1, 0.2],
  ])
  LSTM_ARR_4_2 = np.asarray([
    [-0.1, -0.2],
    [-0.1, -0.2],
    [-0.1, -0.2],
    [-0.1, -0.2],
  ])
  LSTM_ARR_4_1 = np.asarray([
    [-0.1],
    [-0.1],
    [-0.1],
    [-0.1],
  ])
  DEC_LSTM_ARR_8_4 = np.asarray([
    [-0.1, -0.2, -0.1, -0.2],
    [0.1, 0.2, 0.1, 0.2],
    [-0.1, -0.2, -0.1, -0.2],
    [0.1, 0.2, 0.1, 0.2],
    [-0.1, -0.2, -0.1, -0.2],
    [0.1, 0.2, 0.1, 0.2],
    [-0.1, -0.2, -0.1, -0.2],
    [0.1, 0.2, 0.1, 0.2],
  ])


  TYPE_EMB = ([lambda x: x.embeddings], [lambda x: x.embeddings._parameters['weight']])
  TYPE_TRANSFORM = ([lambda x: x.W1, lambda x: x.b1],
                    [lambda x: x.linear._parameters['weight'], lambda x: x.linear._parameters['bias']])
  TYPE_LSTM = ([lambda x: x.Wx[0], lambda x: x.Wh[0], lambda x: x.b[0]],
               [lambda x: x.layers[0]._parameters['weight_ih'], lambda x: x.layers[0]._parameters['weight_hh'], lambda x: x.layers[0]._parameters['bias_ih']])
  TYPE_MLP_ATT = ([lambda x: x.linear_context, lambda x: x.bias_context, lambda x: x.linear_query, lambda x: x.pU],
                  [lambda x: x.linear_context._parameters['weight'], lambda x: x.linear_context._parameters['bias'],
                   lambda x: x.linear_query._parameters['weight'], lambda x: x.pU._parameters['weight']])


  def assert_loss_value(self, desired, rtol=1e-12, atol=0, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    np.testing.assert_allclose(actual=training_regimen.train_loss_tracker.epoch_loss.sum_factors(),
                               desired=desired,
                               rtol=rtol, atol=atol)

  def assert_trained_params(self, desired, rtol=1e-6, atol=0, is_lstm = False,
                                    param_type=TYPE_EMB, subpath="model.src_embedder", flatten=False, *args, **kwargs):
    training_regimen = self.run_training(*args, **kwargs)
    component = training_regimen
    for path_elem in subpath.split("."):
      try:
        component = component[int(path_elem)]
      except ValueError:
        component = getattr(component, path_elem)
    if xnmt.backend_dynet:
      actual = [type_lamb(component).as_array() for type_lamb in param_type[0]]
    else:
      actual = [type_lamb(component).data for type_lamb in param_type[1]]
      if is_lstm: actual = self.convert_pytorch_lstm_weights(actual)
    for sub_param_i, sub_param in enumerate(desired):
      with self.subTest(sub_param_i):
        if flatten:
          np.testing.assert_allclose(actual=actual[sub_param_i].flatten(), desired=sub_param.flatten(), rtol=rtol, atol=atol)
        else:
          np.testing.assert_allclose(actual=actual[sub_param_i], desired=sub_param, rtol=rtol, atol=atol)

  def assert_trained_grads(self, desired, rtol=1e-3, atol=0, flatten=False, param_type=TYPE_EMB, subpath="model.src_embedder",
                               epochs=1, *args, **kwargs):
    assert type(desired) in (list,tuple)
    training_regimen = self.run_training(epochs=epochs-1, *args, **kwargs)
    # last epoch is done manually and without calling update():
    src, trg = next(training_regimen.next_minibatch())
    tt.reset_graph()
    event_trigger.set_train(True)
    loss_builder = training_regimen.training_step(src, trg)
    loss = loss_builder.compute(comb_method=training_regimen.loss_comb_method)
    training_regimen.backward(loss)
    # importantly: no update() here because that would zero out the dynet gradients

    component = training_regimen
    for path_elem in subpath.split("."): component = getattr(component, path_elem)
    if xnmt.backend_dynet:
      actual = [type_lamb(component).grad_as_array() for type_lamb in param_type[0]]
    else:
      actual = [type_lamb(component).grad for type_lamb in param_type[1]]
    for sub_param_i, sub_param in enumerate(desired):
      with self.subTest(sub_param_i):
        if flatten:
          np.testing.assert_allclose(actual=actual[sub_param_i].flatten(), desired=sub_param.flatten(), rtol=rtol,
                                     atol=atol)
        else:
          np.testing.assert_allclose(actual=actual[sub_param_i], desired=sub_param, rtol=rtol, atol=atol)

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
                     tt.npvalue(training_regimen.model.attender.linear_context._parameters['bias'].grad).T, \
                     tt.npvalue(training_regimen.model.attender.pU._parameters['weight'].grad).T,
    np.testing.assert_almost_equal(actual_grads[0], val[0], decimal=places)
    np.testing.assert_almost_equal(actual_grads[1], val[1], decimal=places)
    np.testing.assert_almost_equal(actual_grads[2], val[2], decimal=places)
    np.testing.assert_almost_equal(actual_grads[3].flatten(), val[3].flatten(), decimal=places)

  def convert_pytorch_lstm_weights(self, weights):
    # change ifgo -> ifog; subtract 1-initialized forget gates
    h_dim = weights[0].shape[0] // 4
    return np.concatenate([weights[0][:h_dim * 2,:], weights[0][h_dim * 3:,:],
                           weights[0][h_dim * 2:h_dim * 3, :]], axis=0), \
           np.concatenate([weights[1][:h_dim * 2,:], weights[1][h_dim * 3:,:],
                           weights[1][h_dim * 2:h_dim * 3,:]], axis=0), \
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
    encoder = PyramidalLSTMSeqTransducer(input_dim=layer_dim,
                                         hidden_dim=layer_dim,
                                         downsampling_method='skip',
                                         param_init=InitializerSequence([
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # fwd_l0_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # fwd_l0_hh
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # bwd_l0_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # bwd_l0_hh
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # bwd_l1_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # bwd_l1_hh
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # bwd_l1_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # bwd_l1_hh
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # bwd_l2_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # bwd_l2_hh
                                           InitializerSequence([
                                             NumpyInitializer(self.LSTM_ARR_4_2),  # bwd_l2_ih
                                             NumpyInitializer(self.LSTM_ARR_4_1)]),  # bwd_l2_hh
                                         ]),
                                         layers=3)
    train_args['model'] = \
      DefaultTranslator(
        src_reader=H5Reader(transpose=True, feat_to=2),
        src_embedder=NoopEmbedder(emb_dim=layer_dim),
        encoder=encoder,
        attender=MlpAttender(input_dim=layer_dim, hidden_dim=layer_dim, state_dim=2, param_init=InitializerSequence([NumpyInitializer(self.PROJ_ARR_2_2), NumpyInitializer(self.PROJ_ARR_2_2), NumpyInitializer(self.PROJ_ARR_1_2)])),
        decoder=AutoRegressiveDecoder(
          input_dim=layer_dim,
          embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(self.EMB_ARR_5_2),
                                      fix_norm=1),
          rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                   hidden_dim=layer_dim,
                                   decoder_input_dim=layer_dim,
                                   layers=1,
                                   param_init=InitializerSequence(
                                     [NumpyInitializer(self.DEC_LSTM_ARR_8_4)] + [NumpyInitializer(self.LSTM_ARR_8_2)]),
                                   yaml_path="model.decoder.rnn"),
          transform=NonLinear(input_dim=layer_dim * 2, output_dim=layer_dim, param_init=NumpyInitializer(self.PROJ_ARR_2_4)),
          scorer=Softmax(input_dim=layer_dim, vocab_size=vocab_size, label_smoothing=0.1,
                         param_init=NumpyInitializer(self.EMB_ARR_5_2)),
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

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_basic(self):
    self.assert_loss_value(9.657503128051758)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_three_epochs(self):
    self.assert_loss_value(8.524262428283691, epochs=3, lr=10, rtol=1e-6)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_mlp_att_grads(self):
    desired = (np.asarray([[-3.15510178e-08,-3.91636625e-08],[-6.31020356e-08,-7.83273251e-08]]),
               np.asarray([1.65015179e-09,3.30030359e-09]),
               np.asarray([[-1.15501116e-10,1.41344131e-10],[-2.31002231e-10,2.82688262e-10]]),
               np.asarray([[-1.09920165e-07,1.09920165e-07]]))
    self.assert_trained_grads(desired, epochs=1, param_type=self.TYPE_MLP_ATT, subpath="model.attender", rtol=1e-2)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_mlp_att_grads_two_epochs_adam(self):
    desired = (np.asarray([[1.9886029e-05, 2.2218173e-05], [-5.5424091e-05, -6.2103529e-05]]),
               np.asarray([ 3.0994852e-07, -1.1994176e-05]),
               np.asarray([[-1.9301253e-07, -1.1546918e-08], [7.1632535e-06, 4.7493609e-07]]),
               np.asarray([-1.3820918e-05,  8.8202738e-05]))
    self.assert_trained_grads(desired=desired, epochs=2, adam=True, lr=100, flatten=True,
                              param_type=self.TYPE_MLP_ATT, subpath="model.attender", rtol=1e-2)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_mlp_att_grads_trained(self):
    desired = (np.asarray([[ 6.15145327e-05,5.41474146e-05],[-1.41641664e-04,-1.24026701e-04]]),
               np.asarray([-0.00021929,0.0002702,]),
               np.asarray([[-9.45401825e-06,-1.82743112e-04],[ 1.28936317e-05,2.54007202e-04]]),
               np.asarray([[ 0.00020377,-0.00013509]]))
    self.assert_trained_grads(desired=desired, epochs=3, adam=True, lr=100, param_type=self.TYPE_MLP_ATT, subpath="model.attender")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_three_epochs_adam(self):
    self.assert_loss_value(8.912698745727539, epochs=3, lr=10, adam=True, rtol=1e-6)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_ten_epochs_adam(self):
    self.assert_loss_value(7.670589447021484, epochs=10, lr=10, adam=True)

  #@unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_all_params_one_epoch(self):
    # useful regex: (?<!\[)\s+      ->      ,
    expected = {
      'trg_emb': [np.asarray([[-0.09982011,0.10017989],[-0.2,0.2,],[-0.30009866,0.29990137],[-0.40016636,0.39983365],[-0.5,0.5,]])],
      'transform': (np.asarray([[-0.10014582425355911, -0.1998181790113449 , -0.299998939037323  , -0.39999908208847046],       [ 0.10014582425355911,  0.1998181790113449 ,  0.299998939037323  ,  0.39999908208847046]]),
                    np.asarray([ 0.00010916759492829442, -0.00010916759492829442])),
      'out': (np.asarray([[-0.09950891,0.09950891],[-0.20047086,0.20047086],[-0.30026069,0.30026069],[-0.40025517,0.40025517],[-0.49950439,0.49950439]]),
              np.asarray([-0.05378095,0.03610975,0.03600023,0.03589048,-0.05421948])),
      'enc_l0_fwd': (np.asarray([[-0.09999936819076538, -0.1999993771314621], [-0.0999990701675415, -0.1999991089105606],[-0.09999922662973404, -0.19999924302101135], [-0.09999781101942062, -0.19999800622463226]]),
                     np.asarray(
                       [[-0.10000015795230865], [-0.1000002846121788], [-0.10000022500753403], [-0.10000087320804596]]),
                     np.asarray([-1.9049387844916055e-07, -5.0328594625170808e-07, -3.6109418033447582e-07, -4.4791063373850193e-06])),
      'enc_l0_bwd': (np.asarray([[-0.09999877959489822, -0.19999878108501434],       [-0.099998340010643  , -0.1999983787536621 ],       [-0.09999847412109375, -0.19999852776527405],       [-0.09999625384807587, -0.19999656081199646]]),
                     np.asarray([[-0.1000002771615982 ],       [-0.10000047832727432],       [-0.10000041127204895],       [-0.10000135004520416]]),
                     np.asarray([-3.1626956342734047e-07, -6.8942580355724203e-07, -6.1875067558503360e-07, -8.9420518634142354e-06])),
      'enc_l1_fwd': (np.asarray([[-0.10000033676624298, -0.2000003457069397 ],       [-0.10000044107437134, -0.20000040531158447],       [-0.10000032931566238, -0.2000003159046173 ],       [-0.09999492764472961, -0.19999533891677856]]),
                     np.asarray([[-0.09999993443489075],       [-0.09999991208314896],       [-0.09999993443489075],       [-0.10000104457139969]]),
                     np.asarray([-7.1849507321530837e-07, -9.9630460681510158e-07, -7.2940389372888603e-07,  2.5831532184383832e-05])),
      'enc_l1_bwd': (np.asarray([[-0.10000061243772507, -0.20000059902668   ],       [-0.10000069439411163, -0.20000073313713074],       [-0.10000059753656387, -0.2000006139278412 ],       [-0.09999105334281921, -0.19999200105667114]]),
                     np.asarray([[-0.09999988973140717],       [-0.09999985247850418],       [-0.09999988228082657],       [-0.10000146925449371]]),
                     np.asarray([-1.2661490700338618e-06, -1.4598098232454504e-06, -1.2919989558213274e-06,  5.0705089961411431e-05])),
      'enc_l2_fwd': (np.asarray([[-0.09999994188547134, -0.19999994337558746],       [-0.09999991953372955, -0.19999992847442627],       [-0.09999994188547134, -0.19999994337558746],       [-0.09999449551105499, -0.1999950110912323 ]]),
                     np.asarray([[-0.10000001639127731],       [-0.10000002384185791],       [-0.10000001639127731],       [-0.10000132024288177]]),
                     np.asarray([-6.9932389124005567e-07, -1.0061694410978816e-06, -7.0139543595360010e-07, -1.2505840277299285e-04])),
      'enc_l2_bwd': (np.asarray([[-0.09999989718198776, -0.19999989867210388],       [-0.09999989718198776, -0.1999998837709427 ], [-0.09999991208314896, -0.19999989867210388],       [-0.09999144077301025, -0.199992835521698  ]]),
                     np.asarray([[-0.10000002384185791],       [-0.10000003129243851],       [-0.10000002384185791],       [-0.10000153630971909]]),
                     np.asarray([-1.0538597052800469e-06, -1.1596898730203975e-06, -1.0554830396358739e-06, -2.1529989317059517e-04])),
      'decoder': (np.asarray([[-0.10001152753829956,-0.1999884694814682,-0.0999997928738594,-0.1999998241662979,],[ 0.10002642124891281,0.19997358322143555,0.09999953210353851,0.19999960064888,],[-0.10000862181186676,-0.199991375207901,-0.09999988973140717,-0.19999989867210388],[ 0.1000191867351532,0.19998082518577576,0.09999974817037582,0.19999977946281433],[-0.10001163184642792,-0.19998836517333984,-0.09999974071979523,-0.19999977946281433],[ 0.10002660751342773,0.19997338950634003,0.09999941289424896,0.19999949634075165],[-0.09972044825553894,-0.20027956366539001,-0.10000526160001755,-0.20000450313091278],[ 0.10069657862186432,0.19930341839790344,0.09998748451471329,0.19998928904533386]]),
                  np.asarray([[-0.10000055283308029,-0.1999993473291397,],[ 0.10000120848417282,0.19999858736991882],[-0.10000033676624298,-0.1999996155500412,],[ 0.10000073909759521,0.199999138712883,],[-0.10000085085630417,-0.19999898970127106],[ 0.10000189393758774,0.19999775290489197],[-0.099986232817173,-0.20001618564128876],[ 0.10003256797790527,0.19996173679828644]]),
                  np.asarray([ 1.6304036762448959e-05, -3.7359051930252463e-05,  1.2192504073027521e-05, -2.7129121008329093e-05,  1.6449686881969683e-05, -3.7627756682923064e-05, -3.9535044925287366e-04, -9.8510948009788990e-04])),
      'attender': (np.asarray([[-0.10000000149011612, -0.20000000298023224],       [ 0.10000000894069672,  0.20000001788139343]]),
                    np.asarray([-1.6501518207423516e-10, -3.3003036414847031e-10]),
                    np.asarray([[-0.10000000149011612, -0.20000000298023224],       [ 0.10000000149011612,  0.20000000298023224]]),
                   np.asarray([[-0.09999999403953552, -0.20000001788139343]])),
    }
    self.assert_trained_params(desired=expected['trg_emb'], param_type=self.TYPE_EMB, subpath="model.decoder.embedder")
    self.assert_trained_params(desired=expected['out'], param_type=self.TYPE_TRANSFORM, subpath="model.decoder.scorer.output_projector")
    self.assert_trained_params(desired=expected['transform'], param_type=self.TYPE_TRANSFORM, subpath="model.decoder.transform")
    self.assert_trained_params(desired=expected['attender'], param_type=self.TYPE_MLP_ATT, subpath="model.attender")
    self.assert_trained_params(desired=expected['enc_l0_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.0")
    self.assert_trained_params(desired=expected['enc_l0_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.1")
    self.assert_trained_params(desired=expected['enc_l1_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.0")
    self.assert_trained_params(desired=expected['enc_l1_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.1")
    self.assert_trained_params(desired=expected['enc_l2_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.0")
    self.assert_trained_params(desired=expected['enc_l2_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.1")
    self.assert_trained_params(desired=expected['decoder'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.decoder.rnn")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_all_params_one_epoch_sgd_must_fail(self):
    # useful regex: (?<!\[)\s+      ->      ,
    desired= (np.asarray([-0.09990862756967545, -0.1996491253376007 ,  0.10018278658390045,  0.20070187747478485]),
              np.asarray([1.7103424397646450e-05, 3.4214172046631575e-05]),
              np.asarray([-0.10000356286764145, -0.20000457763671875,  0.09999287128448486,  0.19999085366725922]),
              np.asarray([-0.09920034557580948, -0.20079971849918365]))
    self.assert_trained_params(desired=desired, lr=100, epochs=10, param_type=self.TYPE_MLP_ATT, flatten=True, subpath="model.attender")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads_one_epoch(self):
    desired = [np.asarray(
      [[-0.001798875629901886,-0.001798875629901886,],[ 0.,0.,],[ 0.0009865062311291695,0.0009865062311291695],[ 0.0016636312939226627,0.0016636312939226627],[ 0.,0.,]])]
    self.assert_trained_grads(desired=desired, lr=10, param_type=self.TYPE_EMB, subpath="model.decoder.embedder")

class TestManualBasicSeq2seq(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def run_training(self, num_layers=1, bi_encoder=False, epochs=1, lr=0.1):
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/ab-ba.txt"
    train_args['trg_file'] = "test/data/ab-ba.txt"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    if bi_encoder:
      assert num_layers==1
      encoder = BiLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    param_init=InitializerSequence([InitializerSequence([
                                                                     NumpyInitializer(self.LSTM_ARR_4_2),   # fwd_l0_ih
                                                                     NumpyInitializer(self.LSTM_ARR_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(self.LSTM_ARR_4_2),   # bwd_l0_ih
                                                                     NumpyInitializer(self.LSTM_ARR_4_1)])] # bwd_l0_hh
                                    ),
                                    layers=num_layers)
    else:
      encoder = UniLSTMSeqTransducer(input_dim=layer_dim,
                                     hidden_dim=layer_dim,
                                     param_init=NumpyInitializer(self.LSTM_ARR_8_2),
                                     layers=num_layers)
    train_args['model'] = \
      DefaultTranslator(
        src_reader=PlainTextReader(vocab=vocab),
        trg_reader=PlainTextReader(vocab=vocab),
        src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(self.EMB_ARR_5_2)),
        encoder=encoder,
        attender=DotAttender(),
        decoder=AutoRegressiveDecoder(
          input_dim=layer_dim,
          embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(self.EMB_ARR_5_2)),
          rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                   hidden_dim=layer_dim,
                                   decoder_input_dim=layer_dim,
                                   layers=num_layers,
                                   param_init=InitializerSequence(
                                     [NumpyInitializer(self.DEC_LSTM_ARR_8_4)] + [NumpyInitializer(self.LSTM_ARR_8_2)] * (num_layers*2-1)),
                                   yaml_path="model.decoder.rnn"),
          transform=NonLinear(input_dim=layer_dim * 2, output_dim=layer_dim, param_init=NumpyInitializer(self.PROJ_ARR_2_4)),
          scorer=Softmax(input_dim=layer_dim, vocab_size=vocab_size ,param_init=NumpyInitializer(self.EMB_ARR_5_2)),
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

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_basic(self):
    self.assert_loss_value(9.65715217590332)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_two_epochs(self):
    self.assert_loss_value(6.5872673988342285, epochs=2, lr=10, rtol=1e-7)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_two_layers(self):
    self.assert_loss_value(9.65665054321289, num_layers=2, rtol=1e-7)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_bidirectional(self):
    self.assert_loss_value(9.657083511352539, bi_encoder=True, rtol=1e-7)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_weights_one_epoch(self):
    desired = [np.asarray(
      [[-0.1, 0.1], [-0.20184304, 0.19631392], [-0.30349943, 0.29300117], [-0.40391687, 0.39216626], [-0.5, 0.5]])]
    self.assert_trained_params(desired=desired, lr=100, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-7)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads_one_epoch(self):
    desired = [np.asarray(
      [[0, 0], [1.84304245e-5, 3.68608489e-5], [3.49941438e-5, 6.99882876e-5], [3.91686735e-5, 7.83373471e-5], [0, 0]])]
    self.assert_trained_grads(desired=desired, lr=10, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-6)


  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_all_params_one_epoch(self):
    # TODO: test attender
    expected = {
      'src_emb': [np.asarray([[-0.1, 0.1 ], [-0.2001843, 0.19963139], [-0.30034995, 0.29930013], [-0.4003917, 0.39921662], [-0.5, 0.5]])],
      'trg_emb': [np.asarray([[-0.09716769, 0.10566463], [-0.2, 0.2], [-0.30496135, 0.29007733], [-0.41122547, 0.37754905], [-0.5, 0.5 ]])],
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
    self.assert_trained_params(desired=expected['src_emb'], lr=10, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-6)
    self.assert_trained_params(desired=expected['trg_emb'], lr=10, param_type=self.TYPE_EMB, subpath="model.decoder.embedder", rtol=1e-6)
    self.assert_trained_params(desired=expected['transform'], lr=10, param_type=self.TYPE_TRANSFORM, subpath="model.decoder.transform", rtol=1e-4)
    self.assert_trained_params(desired=expected['out'], lr=10, param_type=self.TYPE_TRANSFORM, subpath="model.decoder.scorer.output_projector", rtol=1e-6)
    self.assert_trained_params(desired=expected['encoder'], lr=10, is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder", rtol=1e-3)
    self.assert_trained_params(desired=expected['decoder'], lr=10, is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.decoder.rnn", rtol=1e-3)

  # TODO: similary, test all gradients

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_weights_two_epochs(self):
    desired = [np.asarray(
      [[-0.1, 0.1], [-0.20195894, 0.19691031], [-0.30414006, 0.29530725], [-0.40498427, 0.3935459], [-0.5, 0.5]])]
    self.assert_trained_params(desired=desired, epochs=2, lr=100, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-4)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads_two_epochs(self):
    desired = [np.asarray(
      [[ 0, 0], [-1.43307407e-05, -2.18112727e-05], [-2.92414807e-05, -4.44276811e-05], [-3.09737370e-05, -4.77675057e-05], [ 0,  0]])]
    self.assert_trained_grads(desired=desired, lr=10, epochs=2, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-4)


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

    if bi_encoder:
      assert num_layers==1
      encoder = BiLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    param_init=InitializerSequence([InitializerSequence([
                                                                     NumpyInitializer(self.LSTM_ARR_4_2),   # fwd_l0_ih
                                                                     NumpyInitializer(self.LSTM_ARR_4_1)]), # fwd_l0_hh
                                                                   InitializerSequence([
                                                                     NumpyInitializer(self.LSTM_ARR_4_2),   # bwd_l0_ih
                                                                     NumpyInitializer(self.LSTM_ARR_4_1)])] # bwd_l0_hh
                                    ),
                                    layers=num_layers)
    else:
      encoder = UniLSTMSeqTransducer(input_dim=layer_dim,
                                     hidden_dim=layer_dim,
                                     param_init=NumpyInitializer(self.LSTM_ARR_8_2),
                                     layers=num_layers)
    train_args['model'] = \
      SequenceClassifier(
        src_reader=PlainTextReader(vocab=vocab),
        trg_reader=IDReader(),
        src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size, param_init=NumpyInitializer(self.EMB_ARR_5_2)),
        encoder=encoder,
        transform=NonLinear(input_dim=layer_dim, output_dim=layer_dim, param_init=NumpyInitializer(self.PROJ_ARR_2_2)),
        scorer=Softmax(input_dim=layer_dim, vocab_size=2 ,param_init=NumpyInitializer(self.OUT_ARR_2_2)),
      )
    train_args['dev_tasks'] = []
    train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = epochs
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_basic(self):
    self.assert_loss_value(1.3862998485565186, rtol=1e-12)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_twolayer(self):
    self.assert_loss_value(1.3862943649291992, num_layers=2, rtol=1e-12)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test__loss_bidirectional(self):
    self.assert_loss_value(1.3863017559051514, bi_encoder=True, rtol=1e-12)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_two_epochs(self):
    self.assert_loss_value(1.3866344690322876, epochs=2, lr=100, rtol=1e-12)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_loss_five_epochs(self):
    self.assert_loss_value(2.6611084938049316, epochs=5, lr=10, rtol=1e-12)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_weights_two_epochs(self):
    desired = [np.asarray(
      [[-0.1, 0.1], [-0.19894804, 0.20147263], [-0.28823119, 0.32002223], [-0.41040528, 0.3818686], [-0.5, 0.5]])]
    self.assert_trained_params(desired=desired, epochs=2, lr=100, param_type=self.TYPE_EMB, subpath="model.src_embedder")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_weights_five_epochs(self):
    expected = [np.asarray(
      [[-0.1, 0.1], [-0.20250981, 0.19391325], [-0.29897961, 0.30119216], [-0.40397269, 0.39145479], [-0.5, 0.5]])]
    self.assert_trained_params(expected, epochs=5, lr=10, param_type=self.TYPE_EMB, subpath="model.src_embedder")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads(self):
    desired = [np.asarray(
      [[0, 0], [1.2468663e-6, 2.49373261e-6], [-5.26151271e-5, -1.05230254e-4], [5.41623740e-5, 1.08324748e-4], [0, 0]])]
    self.assert_trained_grads(desired=desired, param_type=self.TYPE_EMB, subpath="model.src_embedder")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads_two_epochs(self):
    desired = [np.asarray(
      [[ 0, 0], [ 1.23475911e-06, 2.46928539e-06], [-5.26270887e-05, -1.05221523e-04], [ 5.41591871e-05, 1.08285341e-04], [ 0, 0]])]
    self.assert_trained_grads(desired=desired, epochs=2, param_type=self.TYPE_EMB, subpath="model.src_embedder")

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_emb_grads_five_epochs(self):
    desired = [np.asarray(
      [[ 0, 0], [ 1.20434561e-06, 2.40851659e-06], [-5.26594959e-05, -1.05188665e-04], [ 5.41539921e-05, 1.08175940e-04], [ 0, 0]])]
    self.assert_trained_grads(desired=desired, epochs=5 ,param_type=self.TYPE_EMB, subpath="model.src_embedder")


if __name__ == '__main__':
  unittest.main()
