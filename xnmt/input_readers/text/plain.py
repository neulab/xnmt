from typing import Optional

from xnmt import output
from xnmt.vocabs import Vocab
from xnmt.input_readers.text.base import BaseTextReader
from xnmt.sent import SimpleSentence, ScalarSentence
from xnmt.persistence import serializable_init, Serializable


class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files, with one sent per line.

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
    read_sent_len: if set, read the length of each sentence instead of the sentence itself. EOS is not counted.
    output_proc: output processors to revert the created sentences back to a readable string
  """
  yaml_tag = '!PlainTextReader'

  @serializable_init
  def __init__(self, vocab: Optional[Vocab] = None, read_sent_len: bool = False, output_proc = None):
    self.vocab = vocab
    self.read_sent_len = read_sent_len
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  def read_sent(self, line, idx):
    convert_fct = self.vocab.convert if self.vocab else lambda x: int(x)
    if self.read_sent_len:
      return ScalarSentence(idx=idx, value=len(line.strip().split()))
    else:
      try:
        return SimpleSentence(idx=idx,
                              words=[convert_fct(word) for word in line.strip().split()] + [Vocab.ES],
                              vocab=self.vocab,
                              output_procs=self.output_procs)
      except ValueError:
        raise ValueError(f"Expecting integer tokens because no vocab was set. Got: '{x}'")

  def vocab_size(self):
    return len(self.vocab)
