import numbers
from functools import lru_cache

from xnmt import sent
from xnmt.input_readers.input_reader import InputReader


class BaseTextReader(InputReader):

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.Sentence:
    """
    Convert a raw text line into an input object.

    Args:
      line: a single input string
      idx: sentence number
    Returns: a SentenceInput object for the input sentence
    """
    raise RuntimeError("Input readers must implement the read_sent function")

  @lru_cache(maxsize=128)
  def count_sents(self, filename):
    newlines = 0
    with open(filename, 'r+b') as f:
      for _ in f:
        newlines += 1
    return newlines

  def iterate_filtered(self, filename, filter_ids=None):
    """
    Args:
      filename: data file (text file)
      filter_ids:
    Returns: iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield self.read_sent(line=line, idx=sent_count)
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break
