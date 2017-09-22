import math
import xnmt.evaluator as evaluator
import unittest

from xnmt.vocab import Vocab

class TestBLEU(unittest.TestCase):
  def setUp(self):
    self.hyp = ["the taro met the hanako".split()]
    self.ref = ["taro met hanako".split()]

    vocab = Vocab()
    self.hyp_id = list(map(vocab.convert, self.hyp[0]))
    self.ref_id = list(map(vocab.convert, self.ref[0]))

  def test_bleu_1gram(self):
    bleu = evaluator.BLEUEvaluator(ngram=1, smooth=0)
    exp_bleu = 3.0 / 5.0
    act_bleu = bleu.evaluate(self.ref, self.hyp).value()
    self.assertEqual(act_bleu, exp_bleu)

  def test_bleu_4gram_fast(self):
    bleu = evaluator.BLEUEvaluator(ngram=4, smooth=1)
    exp_bleu = math.exp(math.log((3/5) * (2/5) * (1/4) * (1/3))/4)
    act_bleu = bleu.evaluate_fast(self.ref_id, self.hyp_id)
    self.assertEqual(act_bleu, exp_bleu)

if __name__ == '__main__':
  unittest.main()
