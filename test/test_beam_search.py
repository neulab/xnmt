import unittest

import _dynet as dy ; dyparams = dy.DynetParams() ; dyparams.set_random_seed(13); dyparams.init()

import xnmt.model_globals as model_globals
from xnmt.translator import DefaultTranslator
from xnmt.embedder import SimpleWordEmbedder
from xnmt.encoder import LSTMEncoder
from xnmt.attender import StandardAttender
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.training_corpus import BilingualTrainingCorpus
from xnmt.input import BilingualCorpusParser, PlainTextReader

class TestBeamSearch(unittest.TestCase):

  def setUp(self):
    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("some_file", 1)
    self.model = DefaultTranslator(
              src_embedder=SimpleWordEmbedder(vocab_size=100),
              encoder=LSTMEncoder(),
              attender=StandardAttender(),
              trg_embedder=SimpleWordEmbedder(vocab_size=100),
              decoder=MlpSoftmaxDecoder(vocab_size=100),
            )
    self.model.set_train(False)
    self.model.initialize_generator()

    self.training_corpus = BilingualTrainingCorpus(train_src = "examples/data/head.ja",
                                              train_trg = "examples/data/head.en",
                                              dev_src = "examples/data/head.ja",
                                              dev_trg = "examples/data/head.en")
    self.corpus_parser = BilingualCorpusParser(src_reader = PlainTextReader(), 
                                          trg_reader = PlainTextReader())
    self.corpus_parser.read_training_corpus(self.training_corpus)

  def test_scores_improve(self):
    """
    Tests whether beam search improves loss.
    Increasing beam size is not guaranteed to improve the score, but let's at least test
    that beam size 5 is usually better than beam size 1
    """
    better_times = 0
    worse_times = 0
    for sent_id in range(10):
      prev_score = None
      for beam_size in [1,5]:
        dy.renew_cg()
        self.model.initialize_generator(beam=beam_size)
        outputs = self.model.generate_output(self.training_corpus.train_src_data[sent_id], 0)
        output_score = outputs[0][1]
        if prev_score is not None:
          if output_score > prev_score:
            better_times += 1
          elif output_score < prev_score:
            worse_times += 1 
        prev_score = output_score
    self.assertGreater(better_times, worse_times*3)
        



if __name__ == '__main__':
  unittest.main()
