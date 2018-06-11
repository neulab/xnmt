import math
import unittest

import xnmt.evaluator as evaluator
import xnmt.events
from xnmt.test.utils import has_cython
from xnmt.vocab import Vocab

class TestBLEU(unittest.TestCase):
  def setUp(self):
    xnmt.events.clear()
    self.hyp = ["the taro met the hanako".split()]
    self.ref = ["taro met hanako".split()]

    vocab = Vocab()
    self.hyp_id = list(map(vocab.convert, self.hyp[0]))
    self.ref_id = list(map(vocab.convert, self.ref[0]))

  def test_bleu_1gram(self):
    bleu = evaluator.BLEUEvaluator(ngram=1)
    exp_bleu = 3.0 / 5.0
    act_bleu = bleu.evaluate(self.ref, self.hyp).value()
    self.assertEqual(act_bleu, exp_bleu)

  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_bleu_4gram_fast(self):
    bleu = evaluator.FastBLEUEvaluator(ngram=4, smooth=1)
    exp_bleu = math.exp(math.log((3.0/5.0) * (2.0/5.0) * (1.0/4.0) * (1.0/3.0))/4.0)
    act_bleu = bleu.evaluate(self.ref_id, self.hyp_id)
    self.assertEqual(act_bleu, exp_bleu)

class TestGLEU(unittest.TestCase):
  def setUp(self):
    self.evaluator = evaluator.GLEUEvaluator()
  def test_gleu_single_1(self):
    self.assertAlmostEqual(
      self.evaluator.evaluate(['the cat is on the mat'.split()], ['the the the the the the the'.split()]).value(),
      0.0909,
      places=4)
  def test_gleu_single_2(self):
    self.assertAlmostEqual(
      self.evaluator.evaluate(
        ['It is a guide to action that ensures that the military will forever heed Party commands'.split()], [
          'It is a guide to action which ensures that the military always obeys the commands of the party'.split()]).value(),
      0.4393,
      places=3)
  def test_gleu_single_3(self):
    self.assertAlmostEqual(
      self.evaluator.evaluate(
        ['It is a guide to action that ensures that the military will forever heed Party commands'.split()], [
          'It is to insure the troops forever hearing the activity guidebook that party direct'.split()]).value(),
      0.1206,
      places=3)
  def test_gleu_corpus(self):
    self.assertAlmostEqual(
      self.evaluator.evaluate(
        ['It is a guide to action that ensures that the military will forever heed Party commands'.split(),
         'It is a guide to action that ensures that the military will forever heed Party commands'.split()], [
          'It is a guide to action which ensures that the military always obeys the commands of the party'.split(),
          'It is to insure the troops forever hearing the activity guidebook that party direct'.split()]).value(),
      0.2903,
      places=3)

class TestSequenceAccuracy(unittest.TestCase):
  def setUp(self):
    self.evaluator = evaluator.SequenceAccuracyEvaluator()
  def test_correct(self):
    self.assertEqual(self.evaluator.evaluate(["1 2 3".split()], ["1 2 3".split()]).value(), 1.0)
  def test_incorrect(self):
    self.assertEqual(self.evaluator.evaluate(["2 3".split()], ["1 2 3".split()]).value(), 0.0)
  def test_corpus(self):
    self.assertEqual(self.evaluator.evaluate(["1 2 3".split(), "2 3".split()],
                                             ["1 2 3".split(), "1 2 3".split()]).value(),
                     0.5)

if __name__ == '__main__':
  unittest.main()
