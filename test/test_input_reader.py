import unittest

from xnmt import input_reader, sent
import xnmt.vocab

class TestInputReader(unittest.TestCase):

  def test_one_file_multiple_readers(self):
    vocab = xnmt.vocab.Vocab(vocab_file="examples/data/head.en.vocab")
    cr = input_reader.CompoundReader(readers = [input_reader.PlainTextReader(vocab),
                                                input_reader.PlainTextReader(read_sent_len=True)])
    en_sents = list(cr.read_sents(filename="examples/data/head.en"))
    self.assertEqual(len(en_sents), 10)
    self.assertIsInstance(en_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab.i2w[w] for w in en_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(en_sents[0].sents[1].value, len("can you do it in one day ?".split()))

  def test_multiple_files_multiple_readers(self):
    vocab_en = xnmt.vocab.Vocab(vocab_file="examples/data/head.en.vocab")
    vocab_ja = xnmt.vocab.Vocab(vocab_file="examples/data/head.ja.vocab")
    cr = input_reader.CompoundReader(readers = [input_reader.PlainTextReader(vocab_en),
                                                input_reader.PlainTextReader(vocab_ja)])
    mixed_sents = list(cr.read_sents(filename=["examples/data/head.en", "examples/data/head.ja"]))
    self.assertEqual(len(mixed_sents), 10)
    self.assertIsInstance(mixed_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab_en.i2w[w] for w in mixed_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(" ".join([vocab_ja.i2w[w] for w in mixed_sents[0].sents[1].words]), "君 は １ 日 で それ が でき ま す か 。 </s>")


if __name__ == '__main__':
  unittest.main()
