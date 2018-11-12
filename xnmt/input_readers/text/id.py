from xnmt.input_readers.text.base import BaseTextReader
from xnmt.persistence import Serializable, serializable_init
from xnmt.sent import ScalarSentence


class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems).

  Files must be text files containing a single integer per line.
  """
  yaml_tag = "!IDReader"

  @serializable_init
  def __init__(self):
    pass

  def read_sent(self, line, idx):
    return ScalarSentence(idx=idx, value=int(line.strip()))

  def read_sents(self, filename, filter_ids=None):
    return [l for l in self.iterate_filtered(filename, filter_ids)]
