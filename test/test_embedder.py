import unittest

import numpy as np
from itertools import islice

from xnmt.input_reader import PlainTextReader
from xnmt.embedder import PretrainedSimpleWordEmbedder
from xnmt.exp_global import ExpGlobal, NonPersistentParamCollection
import xnmt.events


class PretrainedSimpleWordEmbedderSanityTest(unittest.TestCase):
  def setUp(self):
    xnmt.events.clear()
    self.input_reader = PlainTextReader()
    list(self.input_reader.read_sents('examples/data/head.ja'))
    self.context = ExpGlobal(dynet_param_collection=NonPersistentParamCollection())

  def test_load(self):
    """
    Checks that the embeddings can be loaded, have the right dimension, and that one line matches.
    """
    embedder = PretrainedSimpleWordEmbedder(exp_global=self.context, filename='examples/data/wiki.ja.vec.small', emb_dim=300, vocab=self.input_reader.vocab)
    # self.assertEqual(embedder.embeddings.shape()[::-1], (self.input_reader.vocab_size(), 300))

    with open('examples/data/wiki.ja.vec.small', encoding='utf-8') as vecfile:
      test_line = next(islice(vecfile, 9, None)).split()  # Select the vector for '日'
    test_word = test_line[0]
    test_id = self.input_reader.vocab.w2i[test_word]
    test_emb = test_line[1:]

    self.assertTrue(np.allclose(embedder.embeddings.batch([test_id]).npvalue().tolist(),
                                np.array(test_emb, dtype=float).tolist(), rtol=1e-5))
