import unittest

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt.param_initializers import NumpyInitializer
from xnmt.modelparts.attenders import DotAttender
from xnmt.batchers import SrcBatcher
from xnmt.modelparts.bridges import NoBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
import xnmt.events
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer
from xnmt.loss_calculators import MLELoss
from xnmt.loss_trackers import TrainLossTracker
from xnmt import optimizers
from xnmt.param_collections import ParamManager
from xnmt.train import regimens
from xnmt.modelparts.transforms import NonLinear
from xnmt.models.translators.default import DefaultTranslator
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab
from xnmt import event_trigger, sent



class TestTrainManual(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def test_manual(self):
    layer_dim = 2
    batcher = SrcBatcher(batch_size=2, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/ab-ba.txt"
    train_args['trg_file'] = "test/data/ab-ba.txt"
    train_args['loss_calculator'] = MLELoss()
    vocab = Vocab(i2w=['<s>', '</s>', 'a', 'b', '<unk>'])
    vocab_size = 5
    emb_arr = np.asarray([[-0.1, 0.1],[-0.2, 0.2],[-0.3, 0.3],[-0.4, 0.4],[-0.5, 0.5],])
    proj_arr = np.asarray([
      [-0.1, -0.2, -0.3, -0.4],
      [0.1, 0.2, 0.3, 0.4],
    ])
    lstm_arr = np.asarray([
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
      [-0.1, -0.2],
      [0.1, 0.2],
    ])
    dec_lstm_arr = np.asarray([
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
      [-0.1, -0.2, -0.1, -0.2],
      [0.1, 0.2, 0.1, 0.2],
    ])
    train_args['model'] = DefaultTranslator(src_reader=PlainTextReader(vocab=vocab),
                                            trg_reader=PlainTextReader(vocab=vocab),
                                            src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=vocab_size,
                                                                            param_init=NumpyInitializer(emb_arr)),
                                            encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                                         param_init=NumpyInitializer(lstm_arr)),
                                            attender=DotAttender(),
                                            decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                                                          embedder=SimpleWordEmbedder(emb_dim=layer_dim,
                                                                                                      vocab_size=vocab_size,
                                                                                                      param_init=NumpyInitializer(emb_arr)),
                                                                          rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                                                                   hidden_dim=layer_dim,
                                                                                                   decoder_input_dim=layer_dim,
                                                                                                   param_init=NumpyInitializer(
                                                                                                     lstm_arr),
                                                                                                   yaml_path="model.decoder.rnn"),
                                                                          transform=NonLinear(input_dim=layer_dim * 2,
                                                                                              output_dim=layer_dim,
                                                                                              param_init=NumpyInitializer(proj_arr)),
                                                                          scorer=Softmax(input_dim=layer_dim,
                                                                                         vocab_size=vocab_size),
                                                                          bridge=NoBridge(dec_dim=layer_dim)),
                                            )
    train_args['dev_tasks'] = []
    train_args['trainer'] = optimizers.SimpleSGDTrainer()
    train_args['batcher'] = batcher
    train_args['run_while'] = 'epoch <= 1'
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    from xnmt import trace
    trace.print_trace()
    # self.assertAlmostEqual(training_regimen.train_loss_tracker.epoch_loss.sum_factors() / training_regimen.train_loss_tracker.epoch_words,
    #                        training_regimen.dev_loss_tracker.dev_score.loss, places=5)


if __name__ == '__main__':
  unittest.main()
