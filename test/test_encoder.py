import unittest
import math

import numpy as np
import dynet_config
import dynet as dy

from xnmt.self_attention import MultiHeadAttentionSeqTransducer
from xnmt.pyramidal import PyramidalLSTMSeqTransducer
from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.mlp import MLP
from xnmt.residual import ResidualLSTMSeqTransducer
from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.input_reader import PlainTextReader
from xnmt.param_collection import ParamManager
import xnmt.events
from xnmt.vocab import Vocab

class TestEncoder(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()
    ParamManager.init_param_col()

    self.src_reader = PlainTextReader()
    self.trg_reader = PlainTextReader()
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))

  @xnmt.events.register_xnmt_event
  def set_train(self, val):
    pass
  @xnmt.events.register_xnmt_event
  def start_sent(self, src):
    pass

  def assert_in_out_len_equal(self, model):
    dy.renew_cg()
    self.set_train(True)
    src = self.src_data[0]
    self.start_sent(src)
    embeddings = model.src_embedder.embed_sent(src)
    encodings = model.encoder(embeddings)
    self.assertEqual(len(embeddings), len(encodings))

  def test_bi_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
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
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

  def test_res_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=ResidualLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

  def test_py_lstm_encoder_len(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.set_train(True)
    for sent_i in range(10):
      dy.renew_cg()
      src = self.src_data[sent_i].get_padded_sent(Vocab.ES, 4 - (len(self.src_data[sent_i]) % 4))
      self.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder(embeddings)
      self.assertEqual(int(math.ceil(len(embeddings) / float(4))), len(encodings))

  def test_py_lstm_mask(self):
    layer_dim = 512
    model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=PyramidalLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=1),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )

    batcher = xnmt.batcher.TrgBatcher(batch_size=3)
    train_src, _ = \
      batcher.pack(self.src_data, self.trg_data)

    self.set_train(True)
    for sent_i in range(3):
      dy.renew_cg()
      src = train_src[sent_i]
      self.start_sent(src)
      embeddings = model.src_embedder.embed_sent(src)
      encodings = model.encoder(embeddings)
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
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.assert_in_out_len_equal(model)

if __name__ == '__main__':
  unittest.main()
