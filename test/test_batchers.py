import unittest

from xnmt import batchers, events
from xnmt import sent
from xnmt.param_collections import ParamManager

class TestBatcher(unittest.TestCase):

  def setUp(self):
    events.clear()
    ParamManager.init_param_col()

  def test_batch_src(self):
    src_sents = [sent.SimpleSentence([0] * i, pad_token=1) for i in range(1,7)]
    trg_sents = [sent.SimpleSentence([0] * ((i+3)%6 + 1), pad_token=2) for i in range(1,7)]
    my_batcher = batchers.SrcBatcher(batch_size=3)
    src, trg = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0, 0, 1], [0, 1, 1], [0, 0, 0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]], [x.words for x in trg[1]])

  def test_batch_word_src(self):
    src_sents = [sent.SimpleSentence([0] * i, pad_token=1) for i in range(1,7)]
    trg_sents = [sent.SimpleSentence([0] * ((i+3)%6 + 1), pad_token=2) for i in range(1,7)]
    my_batcher = batchers.WordSrcBatcher(words_per_batch=12)
    src, trg = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 0, 0, 0, 0]], [x.words for x in trg[1]])
    self.assertEqual([[0, 0, 0, 0], [0, 0, 0, 1]], [x.words for x in src[2]])
    self.assertEqual([[0, 0], [0, 2]], [x.words for x in trg[2]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in src[3]])
    self.assertEqual([[0, 0, 0]], [x.words for x in trg[3]])
    self.assertEqual([[0, 0, 0, 0, 0, 0]], [x.words for x in src[4]])
    self.assertEqual([[0, 0, 0, 0]], [x.words for x in trg[4]])

  def test_batch_random_no_ties(self):
    src_sents = [sent.SimpleSentence([0] * i, pad_token=1) for i in range(1,7)]
    trg_sents = [sent.SimpleSentence([0] * ((i+3)%6 + 1), pad_token=2) for i in range(1,7)]
    my_batcher = batchers.SrcBatcher(batch_size=3)
    _, trg = my_batcher.pack(src_sents, trg_sents)
    l0 = trg[0].sent_len()
    for _ in range(10):
      _, trg = my_batcher.pack(src_sents, trg_sents)
      l = trg[0].sent_len()
      self.assertTrue(l==l0)

  def test_batch_random_ties(self):
    src_sents = [sent.SimpleSentence([0] * 5, pad_token=1) for _ in range(1,7)]
    trg_sents = [sent.SimpleSentence([0] * ((i+3)%6 + 1), pad_token=2) for i in range(1,7)]
    my_batcher = batchers.SrcBatcher(batch_size=3)
    _, trg = my_batcher.pack(src_sents, trg_sents)
    l0 = trg[0].sent_len()
    for _ in range(10):
      _, trg = my_batcher.pack(src_sents, trg_sents)
      l = trg[0].sent_len()
      if l!=l0: return
    self.assertTrue(False)

if __name__ == '__main__':
  unittest.main()
