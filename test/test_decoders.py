import unittest

import xnmt
from xnmt.modelparts.attenders import MlpAttender
from xnmt import batchers, event_trigger, events
from xnmt.modelparts.bridges import CopyBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
from xnmt.input_readers import PlainTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.param_collections import ParamManager
from xnmt.modelparts.transforms import NonLinear
from xnmt.models.translators.default import DefaultTranslator
from xnmt.modelparts.scorers import Softmax
from xnmt.search_strategies import GreedySearch
from xnmt.vocabs import Vocab

if xnmt.backend_dynet:
  import dynet as dy

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    events.clear()
    ParamManager.init_param_col()
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(vocab=src_vocab),
      trg_reader=PlainTextReader(vocab=trg_vocab),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
                                rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn"),
                                transform=NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
                                scorer=Softmax(input_dim=layer_dim, vocab_size=100),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  @unittest.skipUnless(xnmt.backend_dynet, "requires dynet backend")
  def test_single(self):
    dy.renew_cg()
    outputs = self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())
    output_score = outputs[0].score

    dy.renew_cg()
    train_loss = self.model.calc_nll(src=self.src_data[0],
                                     trg=outputs[0]).value()

    self.assertAlmostEqual(-output_score, train_loss, places=3)


if __name__ == '__main__':
  unittest.main()
