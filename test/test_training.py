import unittest

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt.modelparts.attenders import MlpAttender, DotAttender
from xnmt.batchers import mark_as_batch, Mask, SrcBatcher
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
from xnmt.eval.tasks import LossEvalTask
import xnmt.events
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculators import MLELoss
from xnmt.loss_trackers import TrainLossTracker
from xnmt import optimizers
from xnmt.param_collections import ParamManager
from xnmt.transducers.pyramidal import PyramidalLSTMSeqTransducer
from xnmt.train import regimens
from xnmt.modelparts.transforms import NonLinear
from xnmt.models.translators.default import DefaultTranslator
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab
from xnmt import event_trigger, sent


class TestTruncatedBatchTraining(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

    self.src_reader = PlainTextReader(vocab=Vocab(vocab_file="test/data/head.ja.vocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="test/data/head.en.vocab"))
    self.src_data = list(self.src_reader.read_sents("test/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("test/data/head.en"))

  def assert_single_loss_equals_batch_loss(self, model, pad_src_to_multiple=1):
    """
    Tests whether single loss equals batch loss.
    Truncating src / trg sents to same length so no masking is necessary
    """
    batch_size=5
    src_sents = self.src_data[:batch_size]
    src_min = min([x.sent_len() for x in src_sents])
    src_sents_trunc = [s.words[:src_min] for s in src_sents]
    for single_sent in src_sents_trunc:
      single_sent[src_min-1] = Vocab.ES
      while len(single_sent)%pad_src_to_multiple != 0:
        single_sent.append(Vocab.ES)
    trg_sents = self.trg_data[:batch_size]
    trg_min = min([x.sent_len() for x in trg_sents])
    trg_sents_trunc = [s.words[:trg_min] for s in trg_sents]
    for single_sent in trg_sents_trunc: single_sent[trg_min-1] = Vocab.ES

    src_sents_trunc = [sent.SimpleSentence(words=s) for s in src_sents_trunc]
    trg_sents_trunc = [sent.SimpleSentence(words=s) for s in trg_sents_trunc]

    single_loss = 0.0
    for sent_id in range(batch_size):
      tt.reset_graph()
      train_loss = MLELoss().calc_loss(
                                   model=model,
                                   src=src_sents_trunc[sent_id],
                                   trg=trg_sents_trunc[sent_id]).value()
      single_loss += train_loss[0]

    tt.reset_graph()

    batched_loss = MLELoss().calc_loss(
                                   model=model,
                                   src=mark_as_batch(src_sents_trunc),
                                   trg=mark_as_batch(trg_sents_trunc)).value()
    self.assertAlmostEqual(single_loss, np.sum(batched_loss), places=4)

  def test_loss_model1(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model, pad_src_to_multiple=4)

  def test_loss_model3(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model4(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=DotAttender(),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

class TestBatchTraining(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

    self.src_reader = PlainTextReader(vocab=Vocab(vocab_file="test/data/head.ja.vocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="test/data/head.en.vocab"))
    self.src_data = list(self.src_reader.read_sents("test/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("test/data/head.en"))

  def assert_single_loss_equals_batch_loss(self, model, pad_src_to_multiple=1):
    """
    Tests whether single loss equals batch loss.
    Here we don't truncate the target side and use masking.
    """
    batch_size = 5
    src_sents = self.src_data[:batch_size]
    src_min = min([x.sent_len() for x in src_sents])
    src_sents_trunc = [s.words[:src_min] for s in src_sents]
    for single_sent in src_sents_trunc:
      single_sent[src_min-1] = Vocab.ES
      while len(single_sent)%pad_src_to_multiple != 0:
        single_sent.append(Vocab.ES)
    trg_sents = sorted(self.trg_data[:batch_size], key=lambda x: x.sent_len(), reverse=True)
    trg_max = max([x.sent_len() for x in trg_sents])
    np_arr = np.zeros([batch_size, trg_max])
    for i in range(batch_size):
      for j in range(trg_sents[i].sent_len(), trg_max):
        np_arr[i,j] = 1.0
    trg_masks = Mask(np_arr)
    trg_sents_padded = [[w for w in s] + [Vocab.ES]*(trg_max-s.sent_len()) for s in trg_sents]

    src_sents_trunc = [sent.SimpleSentence(words=s) for s in src_sents_trunc]
    trg_sents_padded = [sent.SimpleSentence(words=s) for s in trg_sents_padded]

    single_loss = 0.0
    for sent_id in range(batch_size):
      tt.reset_graph()
      train_loss = MLELoss().calc_loss(
                                   model=model,
                                   src=src_sents_trunc[sent_id],
                                   trg=trg_sents[sent_id]).value()
      single_loss += train_loss[0]

    tt.reset_graph()

    batched_loss = MLELoss().calc_loss(
                                   model=model,
                                   src=mark_as_batch(src_sents_trunc),
                                   trg=mark_as_batch(trg_sents_padded, trg_masks)).value()
    self.assertAlmostEqual(single_loss, np.sum(batched_loss), places=4)

  def test_loss_model1(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(layers=3, input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model, pad_src_to_multiple=4)

  def test_loss_model3(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(layers=3, input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

class TestTrainDevLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def test_train_dev_loss_equal(self):
    layer_dim = 512
    batcher = SrcBatcher(batch_size=5, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/head.ja"
    train_args['trg_file'] = "test/data/head.en"
    train_args['loss_calculator'] = MLELoss()
    train_args['model'] = DefaultTranslator(src_reader=PlainTextReader(vocab=Vocab(vocab_file="test/data/head.ja.vocab")),
                                            trg_reader=PlainTextReader(vocab=Vocab(vocab_file="test/data/head.en.vocab")),
                                            src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                            encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
                                            attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim,
                                                                 hidden_dim=layer_dim),
                                            decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                                                      embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                                                      rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                                                                     hidden_dim=layer_dim,
                                                                                                     decoder_input_dim=layer_dim,
                                                                                                     yaml_path="model.decoder.rnn"),
                                                                      transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                                                      scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                                                      bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
                                            )
    train_args['dev_tasks'] = [LossEvalTask(model=train_args['model'],
                                            src_file="test/data/head.ja",
                                            ref_file="test/data/head.en",
                                            batcher=batcher)]
    train_args['trainer'] = optimizers.DummyTrainer()
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = 1
    train_args['train_loss_tracker'] = TrainLossTracker(accumulative=True)
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    training_regimen.run_training(save_fct = lambda: None)
    self.assertAlmostEqual(training_regimen.train_loss_tracker.epoch_loss.sum_factors() / training_regimen.train_loss_tracker.epoch_words,
                           training_regimen.dev_loss_tracker.dev_score.loss, places=5)

class TestOverfitting(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def test_overfitting(self):
    layer_dim = 16
    batcher = SrcBatcher(batch_size=10, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "test/data/head.ja"
    train_args['trg_file'] = "test/data/head.en"
    train_args['loss_calculator'] = MLELoss()
    train_args['model'] = DefaultTranslator(src_reader=PlainTextReader(vocab=Vocab(vocab_file="test/data/head.ja.vocab")),
                                            trg_reader=PlainTextReader(vocab=Vocab(vocab_file="test/data/head.en.vocab")),
                                            src_embedder=SimpleWordEmbedder(vocab_size=100, emb_dim=layer_dim),
                                            encoder=BiLSTMSeqTransducer(input_dim=layer_dim,
                                                                        hidden_dim=layer_dim),
                                            attender=MlpAttender(input_dim=layer_dim,
                                                                 state_dim=layer_dim,
                                                                 hidden_dim=layer_dim),
                                            decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                                                      embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                                                      rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                                                                     hidden_dim=layer_dim,
                                                                                                     decoder_input_dim=layer_dim,
                                                                                                     yaml_path="model.decoder.rnn"),
                                                                      transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                                                      scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                                                      bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
                                            )
    train_args['dev_tasks'] = [LossEvalTask(model=train_args['model'],
                                            src_file="test/data/head.ja",
                                            ref_file="test/data/head.en",
                                            batcher=batcher)]
    train_args['run_for_epochs'] = 1
    train_args['trainer'] = optimizers.AdamTrainer(alpha=0.1, rescale_grads=5.0, skip_noisy=True)

    train_args['batcher'] = batcher
    training_regimen = regimens.SimpleTrainingRegimen(**train_args)
    for _ in range(100):
      training_regimen.run_training(save_fct=lambda:None)
    self.assertAlmostEqual(0.0,
                           training_regimen.train_loss_tracker.epoch_loss.sum_factors() / training_regimen.train_loss_tracker.epoch_words,
                           places=2)

if __name__ == '__main__':
  unittest.main()
