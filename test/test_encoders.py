import unittest
import math

import numpy as np

import xnmt, xnmt.tensor_tools as tt
from xnmt.modelparts.attenders import MlpAttender
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.param_collections import ParamManager
from xnmt.transducers.pyramidal import PyramidalLSTMSeqTransducer
from xnmt.modelparts.scorers import Softmax
from xnmt.transducers.self_attention import MultiHeadAttentionSeqTransducer
from xnmt.modelparts.transforms import NonLinear
from xnmt.models.translators.default import DefaultTranslator
from xnmt.vocabs import Vocab
from xnmt import batchers, event_trigger, events

class TestEncoder(unittest.TestCase):

  def setUp(self):
    events.clear()
    ParamManager.init_param_col()

    src_vocab = Vocab(vocab_file="test/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="test/data/head.en.vocab")
    self.src_reader = PlainTextReader(vocab=src_vocab)
    self.trg_reader = PlainTextReader(vocab=trg_vocab)
    self.src_data = list(self.src_reader.read_sents("test/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("test/data/head.en"))

  def assert_in_out_len_equal(self, model):
    tt.reset_graph()
    event_trigger.set_train(True)
    src = self.src_data[0]
    event_trigger.start_sent(src)
    embeddings = model.src_embedder.embed_sent(src)
    encodings = model.encoder.transduce(embeddings)
    self.assertEqual(len(embeddings), len(encodings))

  def test_bi_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

  def test_uni_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

  # TODO: Update this to the new residual LSTM transducer framework
  # def test_res_lstm_encoder_len(self):
  #   layer_dim = 512
  #   model = DefaultTranslator(
  #     src_reader=self.src_reader,
  #     trg_reader=self.trg_reader,
  #     src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
  #     encoder=ResidualLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
  #     attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
  #     decoder=AutoRegressiveDecoder(input_dim=layer_dim,
  #                               embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
  #                               rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
  #                               transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
  #                               scorer=Softmax(input_dim=layer_dim, vocab_size=100),
  #                               bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
  #   )
  #   self.assert_in_out_len_equal(model)

  def test_py_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(True)
    for sent_i in range(10):
      tt.reset_graph()
      src = self.src_data[sent_i].create_padded_sent(4 - (self.src_data[sent_i].sent_len() % 4))
      event_trigger.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder.transduce(embeddings)
      self.assertEqual(int(math.ceil(len(embeddings) / float(4))), len(encodings))

  def test_py_lstm_mask(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=1),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )

    batcher = batchers.TrgBatcher(batch_size=3)
    train_src, _ = \
      batcher.pack(self.src_data, self.trg_data)

    event_trigger.set_train(True)
    for sent_i in range(3):
      tt.reset_graph()
      src = train_src[sent_i]
      event_trigger.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder.transduce(embeddings)
      if train_src[sent_i].mask is None:
        assert encodings.mask is None
      else:
        np.testing.assert_array_almost_equal(train_src[sent_i].mask.np_arr, encodings.mask.np_arr)

  def test_multihead_attention_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=MultiHeadAttentionSeqTransducer(input_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

if __name__ == '__main__':
  unittest.main()
