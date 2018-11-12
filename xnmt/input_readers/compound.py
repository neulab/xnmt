import numbers
from typing import Sequence, Optional, Iterator, Union

from xnmt.vocabs import Vocab
from xnmt.sent import CompoundSentence, Sentence
from xnmt.input_readers.input_reader import InputReader
from xnmt.persistence import serializable_init, Serializable


class CompoundReader(InputReader, Serializable):
  """
  A compound reader reads inputs using several input readers at the same time.

  The resulting inputs will be of type :class:`CompoundSentence`, which holds the results from the different readers
  as a tuple. Inputs can be read from different locations (if input file name is a sequence of filenames) or all from
  the same location (if it is a string). The latter can be used to read the same inputs using several input different
  readers which might capture different aspects of the input data.

  Args:
    readers: list of input readers to use
    vocab: not used by this reader, but some parent components may require access to the vocab.
  """
  yaml_tag = "!CompoundReader"

  @serializable_init
  def __init__(self, readers:Sequence[InputReader], vocab: Optional[Vocab] = None) -> None:
    assert len(readers) >= 2, "need at least two readers"
    self.readers = readers
    self.vocab = vocab

  def read_sents(self,
                 filename: Union[str,Sequence[str]],
                 filter_ids: Sequence[numbers.Integral] = None) -> Iterator[Sentence]:
    if isinstance(filename, str):
      filename = [filename] * len(self.readers)

    generators = [reader.read_sents(filename=cur_filename, filter_ids=filter_ids) for (reader, cur_filename) in
                  zip(self.readers, filename)]
    while True:
      try:
        sub_sents = tuple([next(gen) for gen in generators])
        yield CompoundSentence(sents=sub_sents)
      except StopIteration:
        return

  def count_sents(self, filename: str) -> int:
    return self.readers[0].count_sents(filename if isinstance(filename, str) else filename[0])

  def needs_reload(self) -> bool:
    return any(reader.needs_reload() for reader in self.readers)
