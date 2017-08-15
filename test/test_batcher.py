import unittest

import xnmt.batcher
import xnmt.input

class TestBatcher(unittest.TestCase):

  def test_batch_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = xnmt.batcher.from_spec("src", 3, src_pad_token=1, trg_pad_token=2)
    src, src_mask, trg, trg_mask = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0, 1, 1], [0, 0, 1], [0, 0, 0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 2, 2], [0, 0, 0, 2], [0, 0, 0, 0]], [x.words for x in trg[1]])

  def test_batch_word_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = xnmt.batcher.from_spec("word_src", 12, src_pad_token=1, trg_pad_token=2)
    src, src_mask, trg, trg_mask = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 1], [0, 0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[1]])
    self.assertEqual([[0, 0, 0, 0]], [x.words for x in src[2]])
    self.assertEqual([[0, 0]], [x.words for x in trg[2]])
    self.assertEqual([[0, 0, 0, 0, 0]], [x.words for x in src[3]])
    self.assertEqual([[0, 0, 0]], [x.words for x in trg[3]])
    self.assertEqual([[0, 0, 0, 0, 0, 0]], [x.words for x in src[4]])
    self.assertEqual([[0, 0, 0, 0]], [x.words for x in trg[4]])

if __name__ == '__main__':
  unittest.main()
