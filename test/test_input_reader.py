import unittest

from xnmt import input_reader
import xnmt.vocab
import xnmt.input

class TestInputReader(unittest.TestCase):

  def test_one_file_multiple_readers(self):
    vocab = xnmt.vocab.Vocab(vocab_file="examples/data/head.en.vocab")
    cr = input_reader.CompoundReader(readers = [input_reader.PlainTextReader(vocab),
                                                input_reader.PlainTextReader(read_sent_len=True)])
    sents = list(cr.read_sents(filename="examples/data/head.en"))
    self.assertEqual(len(sents), 10)
    self.assertIsInstance(sents[0], xnmt.input.CompoundInput)
    self.assertEqual(" ".join([vocab.i2w[w] for w in sents[0].inputs[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(sents[0].inputs[1].value, len("can you do it in one day ?".split()))

  def test_multiple_files_multiple_readers(self):
    vocab_en = xnmt.vocab.Vocab(vocab_file="examples/data/head.en.vocab")
    vocab_ja = xnmt.vocab.Vocab(vocab_file="examples/data/head.ja.vocab")
    cr = input_reader.CompoundReader(readers = [input_reader.PlainTextReader(vocab_en),
                                                input_reader.PlainTextReader(vocab_ja)])
    sents = list(cr.read_sents(filename=["examples/data/head.en", "examples/data/head.ja"]))
    self.assertEqual(len(sents), 10)
    self.assertIsInstance(sents[0], xnmt.input.CompoundInput)
    self.assertEqual(" ".join([vocab_en.i2w[w] for w in sents[0].inputs[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(" ".join([vocab_ja.i2w[w] for w in sents[0].inputs[1].words]), "君 は １ 日 で それ が でき ま す か 。 </s>")


if __name__ == '__main__':
  unittest.main()
