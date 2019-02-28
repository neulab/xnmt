import unittest

import numpy as np
from itertools import islice

import xnmt
from xnmt.input_readers import PlainTextReader
from xnmt.modelparts.embedders import PretrainedSimpleWordEmbedder
from xnmt.param_collections import ParamManager
from xnmt import events
from xnmt.vocabs import Vocab

class PretrainedSimpleWordEmbedderSanityTest(unittest.TestCase):
  def setUp(self):
    events.clear()
    self.input_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.vocab"))
    list(self.input_reader.read_sents('examples/data/head.ja'))
    ParamManager.init_param_col()

  @unittest.skipUnless(xnmt.backend_dynet, "requires dynet backend")
  def test_load(self):
    """
    Checks that the embeddings can be loaded, have the right dimension, and that one line matches.
    """
    embedder = PretrainedSimpleWordEmbedder(filename='examples/data/wiki.ja.vec.small', emb_dim=300, vocab=self.input_reader.vocab)
    # self.assertEqual(embedder.embeddings.shape()[::-1], (self.input_reader.vocab_size(), 300))

    with open('examples/data/wiki.ja.vec.small', encoding='utf-8') as vecfile:
      test_line = next(islice(vecfile, 9, None)).split()  # Select the vector for '日'
    test_word = test_line[0]
    test_id = self.input_reader.vocab.w2i[test_word]
    test_emb = test_line[1:]

    self.assertTrue(np.allclose(embedder.embeddings.batch([test_id]).npvalue().tolist(),
                                np.array(test_emb, dtype=float).tolist(), rtol=1e-5))
