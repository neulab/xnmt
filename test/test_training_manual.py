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

TEST_ALL = False

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

  def assert_trained_grads(self, desired, rtol=1e-3, atol=0, flatten=False, is_lstm=False,
                           param_type=TYPE_EMB, subpath="model.src_embedder", epochs=1, *args, **kwargs):
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
    for path_elem in subpath.split("."):
      try:
        component = component[int(path_elem)]
      except ValueError:
        component = getattr(component, path_elem)
    if xnmt.backend_dynet:
      actual = [type_lamb(component).grad_as_array() for type_lamb in param_type[0]]
    else:
      actual = [type_lamb(component).grad for type_lamb in param_type[1]]
      if is_lstm: actual = self.convert_pytorch_lstm_weights(actual, adjust_forget=False)
    for sub_param_i, sub_param in enumerate(desired):
      with self.subTest(sub_param_i):
        if flatten:
          np.testing.assert_allclose(actual=actual[sub_param_i].flatten(), desired=sub_param.flatten(), rtol=rtol,
                                     atol=atol)
        else:
          np.testing.assert_allclose(actual=actual[sub_param_i], desired=sub_param, rtol=rtol, atol=atol)

  def convert_pytorch_lstm_weights(self, weights, adjust_forget=True):
    # change ifgo -> ifog; subtract 1-initialized forget gates
    h_dim = weights[0].shape[0] // 4
    return np.concatenate([weights[0][:h_dim * 2,:], weights[0][h_dim * 3:,:],
                           weights[0][h_dim * 2:h_dim * 3, :]], axis=0), \
           np.concatenate([weights[1][:h_dim * 2,:], weights[1][h_dim * 3:,:],
                           weights[1][h_dim * 2:h_dim * 3,:]], axis=0), \
           np.concatenate([weights[2][:h_dim], weights[2][h_dim:h_dim * 2] - (1 if adjust_forget else 0),
                           weights[2][h_dim * 3:], weights[2][h_dim * 2:h_dim * 3]], axis=0),

class TestManualFullLAS(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

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

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
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
    self.assert_trained_params(desired=expected['transform'], param_type=self.TYPE_TRANSFORM, subpath="model.decoder.transform", rtol=1e-5)
    self.assert_trained_params(desired=expected['attender'], param_type=self.TYPE_MLP_ATT, subpath="model.attender", rtol=1e-3)
    # all of these LSTM weights have larger difference in the forget gate biases than elsewhere.
    # given that the gradients don't have this characteristic, it's probably just a result of how the "+1" is added to the forget gates
    self.assert_trained_params(desired=expected['enc_l0_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.0", rtol=1e-1)
    self.assert_trained_params(desired=expected['enc_l0_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.1", rtol=1e-1)
    self.assert_trained_params(desired=expected['enc_l1_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.0", rtol=1e-1)
    self.assert_trained_params(desired=expected['enc_l1_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.1", rtol=1e-1)
    self.assert_trained_params(desired=expected['enc_l2_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.0", rtol=1e-1)
    self.assert_trained_params(desired=expected['enc_l2_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.1", rtol=1e-1)
    self.assert_trained_params(desired=expected['decoder'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.decoder.rnn", rtol=1e-1)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_all_grads_one_epoch(self):
    # useful regex: (?<!\[)\s+      ->      ,
    expected = {
      'trg_emb': [np.asarray([[-0.001798875629901886 , -0.001798875629901886 ], [ 0. , 0.], [ 0.0009865062311291695,  0.0009865062311291695], [ 0.0016636312939226627,  0.0016636312939226627], [ 0.,  0.]])],
      'transform': (np.asarray([[ 1.4582253061234951e-03, -1.8182197818532586e-03, -1.0668707545846701e-05, -9.1551919467747211e-06],[-1.4582253061234951e-03,  1.8182197818532586e-03,  1.0668707545846701e-05,  9.1551919467747211e-06]]),
                    np.asarray([-0.0010916759492829442,  0.0010916759492829442])),
      'out': (np.asarray([[-0.004910904914140701 ,0.004910904914140701] ,[ 0.00470859045162797 ,-0.00470859045162797 ,] ,[ 0.00260671554133296 ,-0.00260671554133296 ,] ,[ 0.002551776822656393 ,-0.002551776822656393] ,[-0.00495617650449276 ,0.00495617650449276 ,]]),
              np.asarray([ 0.5378095507621765 , -0.36109745502471924, -0.3600022792816162 , -0.35890477895736694,  0.5421948432922363 ])),
      'enc_l0_fwd': (np.asarray([[-6.3675838646304328e-06, -6.3308766584668774e-06],[-9.3351436589728110e-06, -8.9499744717613794e-06],[-7.7671202234341763e-06, -7.5632206062437035e-06],[-2.1873283913009800e-05, -1.9940496713388711e-05]]),
                     np.asarray([[1.5281908645192743e-06],[2.8453578124754131e-06],[2.2296280803857371e-06],[8.6910986283328384e-06]]),
                     np.asarray([1.9049386992264772e-06, 5.0328594625170808e-06, 3.6109418033447582e-06, 4.4791060645366088e-05])),
      'enc_l0_bwd': (np.asarray([[-1.2237515875312965e-05, -1.2274619621166494e-05], [-1.6632066035526805e-05, -1.6236605006270111e-05],[-1.5294781405827962e-05, -1.4810167158429977e-05],[-3.7490965041797608e-05, -3.4406610211590305e-05]]),
                     np.asarray([[2.7561875413084636e-06],[4.7430157792405225e-06],[4.0998338590725325e-06],[1.3461165508488193e-05]]),
                     np.asarray([3.162695520586567e-06, 6.894258149259258e-06, 6.187506642163498e-06, 8.942051499616355e-05])),
      'enc_l1_fwd': (np.asarray([[ 3.3291255476797232e-06,  3.3630533380346606e-06],[ 4.3786553760583047e-06,  3.9750734686094802e-06],[ 3.2737998481024988e-06,  3.0723647341801552e-06],[-5.0745049520628527e-05, -4.6623012167401612e-05]]),
                     np.asarray([[-6.4850985381781356e-07],[-9.0462390289758332e-07],[-6.6703944412438432e-07],[ 1.0429378562548663e-05]]),
                     np.asarray([ 7.1849503910925705e-06,  9.9630460681510158e-06,  7.2940388236020226e-06, -2.5831532548181713e-04])),
      'enc_l1_bwd': (np.asarray([[ 6.138692242529942e-06,  5.997852895234246e-06],[ 6.943029802641831e-06,  7.248828751471592e-06],[ 5.931152372795623e-06,  6.148908596514957e-06],[-8.950004121288657e-05, -8.007069845916703e-05]]),
                     np.asarray([[-1.1357795983713004e-06],[-1.4717544445375097e-06],[-1.2192750773465377e-06],[ 1.4704629393236246e-05]]),
                     np.asarray([ 1.2661490472964942e-05,  1.4598097550333478e-05,  1.2919989785586949e-05, -5.0705089233815670e-04])),
      'enc_l2_fwd': (np.asarray([[-6.329735242616152e-07, -6.553697744493547e-07],[-8.523718406650005e-07, -7.105951453922899e-07],[-6.071699090171023e-07, -5.338587243386428e-07],[-5.505289664142765e-05, -4.988615182810463e-05]]),
                     np.asarray([[1.3418470246051584e-07],[2.0381197884944413e-07],[1.4089809496908856e-07],[1.3181863323552534e-05]]),
                     np.asarray([6.9932389124005567e-06, 1.0061694410978816e-05, 7.0139540184754878e-06, 1.2505840277299285e-03])),
      'enc_l2_bwd': (np.asarray([[-1.0255213283016928e-06, -9.8833891115646111e-07],[-1.0550540991971502e-06, -1.1603306120377965e-06],[-9.1444701411091955e-07, -1.0104860166393337e-06],[-8.5626517829950899e-05, -7.1609036240261048e-05]]),
                     np.asarray([[2.0159885139037215e-07],[2.6593735924507200e-07],[2.2771446595015732e-07],[1.5319097656174563e-05]]),
                     np.asarray([1.0538597052800469e-05, 1.1596898730203975e-05, 1.0554829714237712e-05, 2.1529989317059517e-03])),
      'decoder': (np.asarray([[ 1.1528694449225441e-04 ,-1.1528694449225441e-04 ,-2.1201644813118037e-06 ,-1.8145493640986388e-06] ,[-2.6416833861730993e-04 ,2.6416833861730993e-04 ,4.6654413381475024e-06 ,3.9940105125424452e-06] ,[ 8.6214015027508140e-05 ,-8.6214015027508140e-05 ,-1.1485036566227791e-06 ,-9.8831151262857020e-07] ,[-1.9183184485882521e-04 ,1.9183184485882521e-04 ,2.5469512365816627e-06 ,2.1922680843999842e-06] ,[ 1.1631683446466923e-04 ,-1.1631683446466923e-04 ,-2.6312920908821980e-06 ,-2.2393367089534877e-06] ,[-2.6606844039633870e-04 ,2.6606844039633870e-04 ,5.8703030845208559e-06 ,4.9977165872405749e-06] ,[-2.7955491095781326e-03 ,2.7955491095781326e-03 ,5.2627845434471965e-05 ,4.5052445784676820e-05] ,[-6.9657745771110058e-03 ,6.9657745771110058e-03 ,1.2515441630966961e-04 ,1.0716813994804397e-04]]),
                  np.asarray([[ 5.5258651627809741e-06 ,-6.4957234826579224e-06] ,[-1.2096515092707705e-05 ,1.4215396731742658e-05] ,[ 3.3449521197326249e-06 ,-3.9367773752019275e-06] ,[-7.4056165431102272e-06 ,8.7145253928611055e-06] ,[ 8.4912808233639225e-06 ,-1.0065879905596375e-05] ,[-1.8960137822432443e-05 ,2.2473535864264704e-05] ,[-1.3768990174867213e-04 ,1.6185421554837376e-04] ,[-3.2564540742896497e-04 ,3.8267995114438236e-04]]),
                  np.asarray([-0.00016304035671055317,  0.0003735905047506094 , -0.00012192504073027521,  0.00027129121008329093, -0.0001644968579057604 ,  0.00037627757410518825,  0.003953504376113415  ,  0.009851094335317612  ])),
      'attender': (np.asarray([[-3.155101779839242e-08, -3.916366253520209e-08],[-6.310203559678484e-08, -7.832732507040419e-08]]),
                    np.asarray([1.650151792986776e-09, 3.300303585973552e-09]),
                    np.asarray([[-1.1550111561620113e-10,  1.4134413084399000e-10], [-2.3100223123240227e-10,  2.8268826168798000e-10]]),
                   np.asarray([[-1.0992016541422345e-07,  1.0992016541422345e-07]])),
    }
    self.assert_trained_grads(desired=expected['trg_emb'], param_type=self.TYPE_EMB, subpath="model.decoder.embedder")
    self.assert_trained_grads(desired=expected['out'], param_type=self.TYPE_TRANSFORM, subpath="model.decoder.scorer.output_projector")
    self.assert_trained_grads(desired=expected['transform'], param_type=self.TYPE_TRANSFORM, subpath="model.decoder.transform", rtol=1e-4)
    self.assert_trained_grads(desired=expected['attender'], param_type=self.TYPE_MLP_ATT, subpath="model.attender", rtol=1e-2)
    self.assert_trained_grads(desired=expected['enc_l0_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.0")
    self.assert_trained_grads(desired=expected['enc_l0_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.0.1")
    self.assert_trained_grads(desired=expected['enc_l1_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.0")
    self.assert_trained_grads(desired=expected['enc_l1_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.1.1")
    self.assert_trained_grads(desired=expected['enc_l2_fwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.0")
    self.assert_trained_grads(desired=expected['enc_l2_bwd'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder.builder_layers.2.1")
    self.assert_trained_grads(desired=expected['decoder'], is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.decoder.rnn")

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

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_all_grads_one_epoch(self):
    expected = {
      'src_emb': [np.asarray([[0.0000000000000000e+00 ,0.0000000000000000e+00] ,[1.8430424461257644e-05 ,3.6860848922515288e-05] ,[3.4994143788935617e-05 ,6.9988287577871233e-05] ,[3.9168673538370058e-05 ,7.8337347076740116e-05] ,[0.0000000000000000e+00 ,0.0000000000000000e+00]])],
      'trg_emb': [np.asarray([[-0.0002832314057741314 ,-0.0005664628115482628] ,[ 0. ,0. ,] ,[ 0.0004961348604410887 ,0.0009922697208821774] ,[ 0.0011225473135709763 ,0.0022450946271419525] ,[ 0. ,0. ,]])],
      'transform': (np.asarray([[ 2.1637452300637960e-03, -2.3929765447974205e-03,  9.0032117441296577e-06, -9.6813309937715530e-06], [-2.1637452300637960e-03,  2.3929765447974205e-03, -9.0032117441296577e-06,  9.6813309937715530e-06]]),
                    np.asarray([-0.0006839334964752197,  0.0006839334964752197])),
      'out': (np.asarray([[-0.003422972746193409 ,0.003422972746193409 ,] ,[ 0.00392539706081152 ,-0.00392539706081152 ,] ,[ 0.0016208074521273375 ,-0.0016208074521273375] ,[ 0.0013080531498417258 ,-0.0013080531498417258] ,[-0.0034312852658331394 ,0.0034312852658331394]]),
              np.asarray([ 1.19862961769104 ,-0.8006855249404907 ,-0.8000003695487976 ,-0.7993147373199463 ,1.2013713121414185])),
      'encoder': (np.asarray([[ 4.6908144213375635e-06 ,-4.6908144213375635e-06] ,[-9.1225047071930021e-06 ,9.1225047071930021e-06] ,[ 1.0857911547645926e-06 ,-1.0857911547645926e-06] ,[-2.1115356503287330e-06 ,2.1115356503287330e-06] ,[ 4.0194663597503677e-06 ,-4.0194663597503677e-06] ,[-7.8346638474613428e-06 ,7.8346638474613428e-06] ,[-2.6696710847318172e-04 ,2.6696710847318172e-04] ,[-5.3662038408219814e-04 ,5.3662038408219814e-04]]),
                  np.asarray([[ 7.0258742823625653e-08 ,-7.5740778981980839e-08] ,[-1.3502631190931424e-07 ,1.4556835026269255e-07] ,[ 4.7886871357150085e-08 ,-5.1737405470930753e-08] ,[-9.3461657968418876e-08 ,1.0097793534669108e-07] ,[ 1.2640501267924265e-07 ,-1.3651916219714622e-07] ,[-2.4672652898516390e-07 ,2.6646509354577574e-07] ,[-4.9538684834260494e-06 ,5.3469202612177469e-06] ,[-9.7866331998375244e-06 ,1.0562853276496753e-05]]),
                  np.asarray([-1.3880884580430575e-05,  2.6948131562676281e-05, -4.0792547224555165e-06,  7.9521041698171757e-06, -1.3859059436072130e-05,  2.7024401788366958e-05,  8.3017250290140510e-04,  1.6623610863462090e-03])),
      'decoder': (np.asarray([[ 8.9689070591703057e-05 ,-8.9689070591703057e-05 ,3.2088462376123061e-06 ,-3.4504373616073281e-06] ,[-1.8882330914493650e-04 ,1.8882330914493650e-04 ,-6.7475839387043379e-06 ,7.2556249506305903e-06] ,[ 3.1497544114245102e-05 ,-3.1497544114245102e-05 ,1.1835369377877214e-06 ,-1.2726726481560036e-06] ,[-6.5534200984984636e-05 ,6.5534200984984636e-05 ,-2.4655400920892134e-06 ,2.6512284421187360e-06] ,[ 9.1197827714495361e-05 ,-9.1197827714495361e-05 ,3.3370115488651209e-06 ,-3.5882007978216279e-06] ,[-1.9217205408494920e-04 ,1.9217205408494920e-04 ,-7.0300829975167289e-06 ,7.5592788562062196e-06] ,[-4.1559892706573009e-03 ,4.1559892706573009e-03 ,-1.5928641369100660e-04 ,1.7127458704635501e-04] ,[-9.1924378648400307e-03 ,9.1924378648400307e-03 ,-3.4886479261331260e-04 ,3.7512285052798688e-04]]),
                  np.asarray([[ 2.0806178326893132e-06 ,-2.2286344574240502e-06] ,[-4.3365957935748156e-06 ,4.6441164158750325e-06] ,[ 9.2460328460219898e-07 ,-9.9547685294965049e-07] ,[-1.9243170754634775e-06 ,2.0718568976008100e-06] ,[ 3.4002866868831916e-06 ,-3.6713533972942969e-06] ,[-7.1314802880806383e-06 ,7.6995174822513945e-06] ,[-1.0815544374054298e-04 ,1.1607046326389536e-04] ,[-2.3403797240462154e-04 ,2.5108811678364873e-04]]),
                  np.asarray([-2.2810854716226459e-04,  4.8337105545215309e-04, -8.9944369392469525e-05,  1.8737521895673126e-04, -2.2850495588500053e-04,  4.8358712228946388e-04,  9.0061575174331665e-03,  2.0659774541854858e-02])),
    }
    self.assert_trained_grads(desired=expected['src_emb'], lr=10, param_type=self.TYPE_EMB, subpath="model.src_embedder", rtol=1e-6)
    self.assert_trained_grads(desired=expected['trg_emb'], lr=10, param_type=self.TYPE_EMB, subpath="model.decoder.embedder", rtol=1e-6)
    self.assert_trained_grads(desired=expected['transform'], lr=10, param_type=self.TYPE_TRANSFORM, subpath="model.decoder.transform", rtol=1e-4)
    self.assert_trained_grads(desired=expected['out'], lr=10, param_type=self.TYPE_TRANSFORM, subpath="model.decoder.scorer.output_projector", rtol=1e-6)
    self.assert_trained_grads(desired=expected['encoder'], lr=10, is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.encoder", rtol=1e-3)
    self.assert_trained_grads(desired=expected['decoder'], lr=10, is_lstm=True, param_type=self.TYPE_LSTM, subpath="model.decoder.rnn", rtol=1e-3)


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



class TestGradientClippingManual(unittest.TestCase, ManualTestingBaseClass):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def run_training(self, lr=0.1, rescale_grads=None):
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/ab-ba.txt"
    train_args['trg_file'] = "test/data/ab-ba.txt"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    encoder = UniLSTMSeqTransducer(input_dim=layer_dim,
                                   hidden_dim=layer_dim,
                                   param_init=NumpyInitializer(self.LSTM_ARR_8_2),
                                   layers=1)
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
                                   layers=1,
                                   param_init=InitializerSequence(
                                     [NumpyInitializer(self.DEC_LSTM_ARR_8_4)] + [NumpyInitializer(self.LSTM_ARR_8_2)]),
                                   yaml_path="model.decoder.rnn"),
          transform=NonLinear(input_dim=layer_dim * 2, output_dim=layer_dim, param_init=NumpyInitializer(self.PROJ_ARR_2_4)),
          scorer=Softmax(input_dim=layer_dim, vocab_size=vocab_size ,param_init=NumpyInitializer(self.EMB_ARR_5_2)),
          bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
      )
    train_args['dev_tasks'] = []
    if xnmt.backend_dynet:
      train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=rescale_grads or -1)
    else:
      train_args['trainer'] = optimizers.SimpleSGDTrainer(e0=lr, clip_grads=0, rescale_grads=rescale_grads or 0)
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = 0
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    # training_regimen.run_training(save_fct = lambda: None)
    return training_regimen

  def assert_trained_grads(self, desired, rtol=1e-3, atol=0, flatten=False, is_lstm=False,
                           param_type=ManualTestingBaseClass.TYPE_EMB, subpath="model.src_embedder", *args, **kwargs):
    assert type(desired) in (list,tuple)
    training_regimen = self.run_training(*args, **kwargs)
    # last epoch is done manually and without calling update():
    src, trg = next(training_regimen.next_minibatch())
    tt.reset_graph()
    event_trigger.set_train(True)
    loss_builder = training_regimen.training_step(src, trg)
    loss = loss_builder.compute(comb_method=training_regimen.loss_comb_method)
    loss = 1000 * loss
    training_regimen.backward(loss)
    training_regimen.update(training_regimen.trainer)

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
      if is_lstm: actual = self.convert_pytorch_lstm_weights(actual, adjust_forget=False)

    for sub_param_i, sub_param in enumerate(desired):
      with self.subTest(sub_param_i):
        if flatten:
          np.testing.assert_allclose(actual=actual[sub_param_i].flatten(), desired=sub_param.flatten(), rtol=rtol,
                                     atol=atol)
        else:
          np.testing.assert_allclose(actual=actual[sub_param_i], desired=sub_param, rtol=rtol, atol=atol)

  @unittest.skipUnless(TEST_ALL, reason="quick subtest")
  def test_no_clipping(self):
    desired = [np.asarray(
      [[-0.10000000149011612, 0.10000000149011612],
       [-0.2018430382013321, 0.19631394743919373],
       [-0.30349940061569214, 0.2930012345314026],
       [-0.4039168357849121, 0.3921663165092468],
       [-0.5, 0.5]])]
    self.assert_trained_grads(desired=desired, rescale_grads=None, param_type=self.TYPE_EMB, subpath="model.src_embedder")

  def test_clipping(self):
    desired = [np.asarray(
      [[-0.10000000149011612,0.10000000149011612],
       [-0.2000042051076889,0.1999915987253189,],
       [-0.30000799894332886,0.29998403787612915],
       [-0.4000089466571808,0.39998212456703186],
       [-0.5,0.5,]])]
    self.assert_trained_grads(desired=desired, rescale_grads=5, param_type=self.TYPE_EMB, subpath="model.src_embedder")


if __name__ == '__main__':
  unittest.main()
