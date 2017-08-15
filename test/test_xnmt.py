import sys
import dynet as dy
import random
import unittest

import xnmt.batcher as batcher
import xnmt.input
#import xnmt.specialized_encoders as specialized_encoders
#import xnmt.expression_sequence as expression_sequence
#import xnmt.model_globals as model_globals

class TestBatcher(unittest.TestCase):

  def test_batch_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = batcher.from_spec("src", 3, src_pad_token=1, trg_pad_token=2)
    src, src_mask, trg, trg_mask = my_batcher.pack(src_sents, trg_sents)
    self.assertEqual([[0, 1, 1], [0, 0, 1], [0, 0, 0]], [x.words for x in src[0]])
    self.assertEqual([[0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]], [x.words for x in trg[0]])
    self.assertEqual([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], [x.words for x in src[1]])
    self.assertEqual([[0, 0, 2, 2], [0, 0, 0, 2], [0, 0, 0, 0]], [x.words for x in trg[1]])

  def test_batch_word_src(self):
    src_sents = [xnmt.input.SimpleSentenceInput([0] * i) for i in range(1,7)]
    trg_sents = [xnmt.input.SimpleSentenceInput([0] * ((i+3)%6 + 1)) for i in range(1,7)]
    my_batcher = batcher.from_spec("word_src", 12, src_pad_token=1, trg_pad_token=2)
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

class TestRunningConfig(unittest.TestCase):
  def test_debug_yaml(self):
    import xnmt.xnmt_run_experiments as xnmt_run_experiments
    xnmt_run_experiments.main(["examples/debug.yaml"])
  
# TODO: fix or delete test
#class TestSpecializedEncoders(unittest.TestCase):
#
#  def test_harwath_speech_works_on_short_inputs(self):
#    model_globals.dynet_param_collection = model_globals.PersistentParamCollection("/tmp/model", 1)
#
#    enc = specialized_encoders.HarwathSpeechEncoder(
#                 filter_height = [2, 1, 1],
#                 filter_width = [5, 25, 25],
#                 channels = [1, 4, 6],
#                 num_filters = [4, 6, 8],
#                 stride= [1, 1, 1])
#    expseq = expression_sequence.ExpressionSequence(expr_list = [dy.zeroes([2])])
#    enc.transduce(expseq)

if __name__ == '__main__':
  unittest.main()
