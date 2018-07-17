import unittest

import dynet as dy
import numpy as np

from xnmt.attender import MlpAttender, DotAttender
from xnmt.batcher import mark_as_batch, Mask, SrcBatcher
from xnmt.bridge import CopyBridge
from xnmt.decoder import AutoRegressiveDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.eval_task import LossEvalTask
import xnmt.events
from xnmt.input_reader import PlainTextReader, SimpleSentenceInput
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculator import AutoRegressiveMLELoss
from xnmt.optimizer import AdamTrainer, DummyTrainer
from xnmt.param_collection import ParamManager
from xnmt.pyramidal import PyramidalLSTMSeqTransducer
import xnmt.training_regimen
from xnmt.transform import NonLinear
from xnmt.translator import DefaultTranslator
from xnmt.scorer import Softmax
from xnmt.vocab import Vocab

class TestTruncatedBatchTraining(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

    self.src_reader = PlainTextReader()
    self.trg_reader = PlainTextReader()
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))

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

    src_sents_trunc = [SimpleSentenceInput(s) for s in src_sents_trunc]
    trg_sents_trunc = [SimpleSentenceInput(s) for s in trg_sents_trunc]

    single_loss = 0.0
    for sent_id in range(batch_size):
      dy.renew_cg()
      train_loss = model.calc_loss(src=src_sents_trunc[sent_id],
                                   trg=trg_sents_trunc[sent_id],
                                   loss_calculator=AutoRegressiveMLELoss()).value()
      single_loss += train_loss

    dy.renew_cg()

    batched_loss = model.calc_loss(src=mark_as_batch(src_sents_trunc),
                                   trg=mark_as_batch(trg_sents_trunc),
                                   loss_calculator=AutoRegressiveMLELoss()).value()
    self.assertAlmostEqual(single_loss, np.sum(batched_loss), places=4)

  def test_loss_model1(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model, pad_src_to_multiple=4)

  def test_loss_model3(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model4(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=DotAttender(),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

class TestBatchTraining(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

    self.src_reader = PlainTextReader()
    self.trg_reader = PlainTextReader()
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))

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

    src_sents_trunc = [SimpleSentenceInput(s) for s in src_sents_trunc]
    trg_sents_padded = [SimpleSentenceInput(s) for s in trg_sents_padded]

    single_loss = 0.0
    for sent_id in range(batch_size):
      dy.renew_cg()
      train_loss = model.calc_loss(src=src_sents_trunc[sent_id],
                                   trg=trg_sents[sent_id],
                                   loss_calculator=AutoRegressiveMLELoss()).value()
      single_loss += train_loss

    dy.renew_cg()

    batched_loss = model.calc_loss(src=mark_as_batch(src_sents_trunc),
                                   trg=mark_as_batch(trg_sents_padded, trg_masks),
                                   loss_calculator=AutoRegressiveMLELoss()).value()
    self.assertAlmostEqual(single_loss, np.sum(batched_loss), places=4)

  def test_loss_model1(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)

  def test_loss_model2(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(layers=3, input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model, pad_src_to_multiple=4)

  def test_loss_model3(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(layers=3, input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                               hidden_dim=layer_dim,
                                                               decoder_input_dim=layer_dim,
                                                               yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    model.set_train(False)
    self.assert_single_loss_equals_batch_loss(model)


class TestTrainDevLoss(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

  def test_train_dev_loss_equal(self):
    layer_dim = 512
    batcher = SrcBatcher(batch_size=5, break_ties_randomly=False)
    train_args = {}
    train_args['src_file'] = "examples/data/head.ja"
    train_args['trg_file'] = "examples/data/head.en"
    train_args['loss_calculator'] = AutoRegressiveMLELoss()
    train_args['model'] = DefaultTranslator(src_reader=PlainTextReader(),
                                            trg_reader=PlainTextReader(),
                                            src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                            encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
                                            attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim,
                                                                 hidden_dim=layer_dim),
                                            trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                            decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                                                      trg_embed_dim=layer_dim,
                                                                      rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                                                                     hidden_dim=layer_dim,
                                                                                                     decoder_input_dim=layer_dim,
                                                                                                     yaml_path="model.decoder.rnn"),
                                                                      transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                                                      scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                                                      bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
                                            )
    train_args['dev_tasks'] = [LossEvalTask(model=train_args['model'],
                                            src_file="examples/data/head.ja",
                                            ref_file="examples/data/head.en",
                                            batcher=batcher)]
    train_args['trainer'] = DummyTrainer()
    train_args['batcher'] = batcher
    train_args['run_for_epochs'] = 1
    training_regimen = xnmt.training_regimen.SimpleTrainingRegimen(**train_args)
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
    train_args['src_file'] = "examples/data/head.ja"
    train_args['trg_file'] = "examples/data/head.en"
    train_args['loss_calculator'] = AutoRegressiveMLELoss()
    train_args['model'] = DefaultTranslator(src_reader=PlainTextReader(),
                                            trg_reader=PlainTextReader(),
                                            src_embedder=SimpleWordEmbedder(vocab_size=100, emb_dim=layer_dim),
                                            encoder=BiLSTMSeqTransducer(input_dim=layer_dim,
                                                                        hidden_dim=layer_dim),
                                            attender=MlpAttender(input_dim=layer_dim,
                                                                 state_dim=layer_dim,
                                                                 hidden_dim=layer_dim),
                                            trg_embedder=SimpleWordEmbedder(vocab_size=100, emb_dim=layer_dim),
                                            decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                                                      trg_embed_dim=layer_dim,
                                                                      rnn=UniLSTMSeqTransducer(input_dim=layer_dim,
                                                                                                     hidden_dim=layer_dim,
                                                                                                     decoder_input_dim=layer_dim,
                                                                                                     yaml_path="model.decoder.rnn"),
                                                                      transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                                                      scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                                                      bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
                                            )
    train_args['dev_tasks'] = [LossEvalTask(model=train_args['model'],
                                            src_file="examples/data/head.ja",
                                            ref_file="examples/data/head.en",
                                            batcher=batcher)]
    train_args['run_for_epochs'] = 1
    train_args['trainer'] = AdamTrainer(alpha=0.1)
    train_args['batcher'] = batcher
    training_regimen = xnmt.training_regimen.SimpleTrainingRegimen(**train_args)
    for _ in range(50):
      training_regimen.run_training(save_fct=lambda:None)
    self.assertAlmostEqual(0.0,
                           training_regimen.train_loss_tracker.epoch_loss.sum_factors() / training_regimen.train_loss_tracker.epoch_words,
                           places=2)

if __name__ == '__main__':
  unittest.main()
