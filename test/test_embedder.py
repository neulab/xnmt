# -*- coding: utf-8 -*-

import unittest

import io
import numpy as np
from itertools import islice

from xnmt.input import PlainTextReader
from xnmt.embedder import PretrainedSimpleWordEmbedder
from xnmt.model_context import ModelContext, PersistentParamCollection
import xnmt.events


class PretrainedSimpleWordEmbedderSanityTest(unittest.TestCase):
  def setUp(self):
    xnmt.events.clear()
    self.input_reader = PlainTextReader()
    list(self.input_reader.read_sents('examples/data/head.ja'))
    self.input_reader.freeze()
    self.context = ModelContext()
    self.context.dynet_param_collection = PersistentParamCollection(None, 0)

  def test_load(self):
    """
    Checks that the embeddings can be loaded, have the right dimension, and that one line matches.
    """
    embedder = PretrainedSimpleWordEmbedder(self.context, self.input_reader.vocab, 'examples/data/wiki.ja.vec.small', 300)
    self.assertEqual(embedder.embeddings.shape()[::-1], (self.input_reader.vocab_size(), 300))

    with io.open('examples/data/wiki.ja.vec.small', encoding='utf-8') as vecfile:
      test_line = next(islice(vecfile, 9, None)).split()  # Select the vector for '日'
    test_word = test_line[0]
    test_id = self.input_reader.vocab.w2i[test_word]
    test_emb = test_line[1:]

    self.assertTrue(np.allclose(embedder.embeddings.batch([test_id]).npvalue().tolist(),
                                np.array(test_emb, dtype=float).tolist(), rtol=1e-5))
