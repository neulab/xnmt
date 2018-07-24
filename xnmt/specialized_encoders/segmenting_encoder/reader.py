from xnmt.input_reader import PlainTextReader
from xnmt.persistence import serializable_init, Serializable
from xnmt.vocab import Vocab
from xnmt.sent import SimpleSentence

class CharFromWordTextReader(PlainTextReader, Serializable):
  yaml_tag = "!CharFromWordTextReader"
  @serializable_init
  def __init__(self, vocab=None):
    super().__init__(vocab)
  def read_sent(self, line, idx):
    chars = []
    segs = []
    offset = 0
    for word in line.strip().split():
      offset += len(word)
      segs.append(offset-1)
      chars.extend([c for c in word])
    segs.append(len(chars))
    chars.append(Vocab.ES_STR)
    sent_input = SimpleSentence(words=[self.vocab.convert(c) for c in chars], idx=idx)
    sent_input.segment = segs
    return sent_input
