import unittest

import _dynet as dy ; dyparams = dy.DynetParams() ; dyparams.set_random_seed(13); dyparams.init()

import xnmt.model_globals as model_globals
from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.encoder import LSTMEncoder, PyramidalLSTMEncoder
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder, CopyBridge
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader
from xnmt.batcher import mark_as_batch
import xnmt.xnmt_train
from xnmt.options import Args

class TestBatchTraining(unittest.TestCase):
  
  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(), 
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def assert_single_loss_equals_batch_loss(self, model, batch_size=5):
    """
    Tests whether single loss equals batch loss.
    Truncating src / trg sents to same length so no masking is necessary 
    """
    batch_size = 5
    src_sents = self.training_corpus.train_src_data[:batch_size]
    src_min = min([len(x) for x in src_sents])
    src_sents_trunc = [s[:src_min] for s in src_sents]
    trg_sents = self.training_corpus.train_trg_data[:batch_size]
    trg_min = min([len(x) for x in trg_sents])
    trg_sents_trunc = [s[:trg_min] for s in trg_sents]
    
    single_loss = 0.0
    for sent_id in range(batch_size):
      dy.renew_cg()
      train_loss = model.calc_loss(src=src_sents_trunc[sent_id], 
                                        trg=trg_sents_trunc[sent_id],
                                        src_mask=None, trg_mask=None).value()
      single_loss += train_loss
    
    dy.renew_cg()
    
    batched_loss = model.calc_loss(src=mark_as_batch(src_sents_trunc), 
                                        trg=mark_as_batch(trg_sents_trunc),
                                        src_mask=None, trg_mask=None).value()
    self.assertAlmostEqual(single_loss, sum(batched_loss), places=4)
  
  def test_loss_model1(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=PyramidalLSTMEncoder(layers=3),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model3(self):
    model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(layers=3),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100, bridge=CopyBridge(dec_layers=1)),
            )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)
    
    
class TestTrainDevLoss(unittest.TestCase):
  
  def test_train_dev_loss_equal(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection(None, 0)
    task_options = xnmt.xnmt_train.options
    train_args = dict({opt.name: opt.default_value for opt in task_options if
                                opt.default_value is not None or not opt.required})
    train_args['training_corpus'] = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                                            train_trg = "examples/data/head.en",
                                                            dev_src = "examples/data/head.ja",
                                                            dev_trg = "examples/data/head.en")
    train_args['corpus_parser'] = BilingualCorpusParser(src_reader = PlainTextReader(), 
                                                        trg_reader = PlainTextReader())
    train_args['model'] = DefaultTranslator(src_embedder=SimpleWordEmbedder(vocab_size=100),
                                            encoder=LSTMEncoder(),
                                            attender=StandardAttender(),
                                            trg_embedder=SimpleWordEmbedder(vocab_size=100),
                                            decoder=MlpSoftmaxDecoder(vocab_size=100),
                                            )
    train_args['model_file'] = None
    train_args['save_num_checkpoints'] = 0
    xnmt_trainer = xnmt.xnmt_train.XnmtTrainer(args=Args(**train_args), need_deserialization=False)
    xnmt_trainer.run_epoch(update_weights=False)
    self.assertAlmostEqual(xnmt_trainer.logger.epoch_loss.loss_values['loss'] / xnmt_trainer.logger.epoch_words,
                           xnmt_trainer.logger.dev_score.loss)

class TestOverfitting(unittest.TestCase):
  
  def test_overfitting(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection(None, 0)
    model_globals.model_globals["default_layer_dim"] = 16
    task_options = xnmt.xnmt_train.options
    train_args = dict({opt.name: opt.default_value for opt in task_options if
                                opt.default_value is not None or not opt.required})
    train_args['training_corpus'] = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                                            train_trg = "examples/data/head.en",
                                                            dev_src = "examples/data/head.ja",
                                                            dev_trg = "examples/data/head.en")
    train_args['corpus_parser'] = BilingualCorpusParser(src_reader = PlainTextReader(), 
                                                        trg_reader = PlainTextReader())
    train_args['model'] = DefaultTranslator(src_embedder=SimpleWordEmbedder(vocab_size=100),
                                            encoder=LSTMEncoder(),
                                            attender=StandardAttender(),
                                            trg_embedder=SimpleWordEmbedder(vocab_size=100),
                                            decoder=MlpSoftmaxDecoder(vocab_size=100),
                                            )
    train_args['model_file'] = None
    train_args['save_num_checkpoints'] = 0
    xnmt_trainer = xnmt.xnmt_train.XnmtTrainer(args=Args(**train_args), need_deserialization=False)
    for _ in range(1000):
      xnmt_trainer.run_epoch(update_weights=True)
    self.assertAlmostEqual(0.0, 
                           xnmt_trainer.logger.epoch_loss.loss_values['loss'] / xnmt_trainer.logger.epoch_words,
                           places=2)

if __name__ == '__main__':
  unittest.main()
