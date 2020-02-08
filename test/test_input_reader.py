import unittest

from xnmt import input_readers, sent
from xnmt import vocabs

class TestInputReader(unittest.TestCase):

  def test_one_file_multiple_readers(self):
    vocab = vocabs.Vocab(vocab_file="test/data/head.en.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab),
                                               input_readers.PlainTextReader(read_sent_len=True)])
    en_sents = list(cr.read_sents(filename="test/data/head.en"))
    self.assertEqual(len(en_sents), 10)
    self.assertIsInstance(en_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab.i2w[w] for w in en_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(en_sents[0].sents[1].value, len("can you do it in one day ?".split()))

  def test_multiple_files_multiple_readers(self):
    vocab_en = vocabs.Vocab(vocab_file="test/data/head.en.vocab")
    vocab_ja = vocabs.Vocab(vocab_file="test/data/head.ja.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab_en),
                                               input_readers.PlainTextReader(vocab_ja)])
    mixed_sents = list(cr.read_sents(filename=["test/data/head.en", "test/data/head.ja"]))
    self.assertEqual(len(mixed_sents), 10)
    self.assertIsInstance(mixed_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab_en.i2w[w] for w in mixed_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(" ".join([vocab_ja.i2w[w] for w in mixed_sents[0].sents[1].words]), "君 は １ 日 で それ が でき ま す か 。 </s>")


class TestCoNLLInputReader(unittest.TestCase):
  
  def test_read_tree(self):
    vocab = vocabs.Vocab(vocab_file="test/data/dep_tree.vocab")
    reader = input_readers.CoNLLToRNNGActionsReader(vocab, vocab)
    tree = list(reader.read_sents(filename="test/data/dep_tree.conll"))
    expected = [sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("David")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("Gallo")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, True),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert(":")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, False),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("This")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("is")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("Bill")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("Lange")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, True),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, True),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, True),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, False),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert(".")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE, False)]
    self.assertListEqual(tree[0].actions, expected)
  

if __name__ == '__main__':
  unittest.main()
