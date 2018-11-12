from xnmt.input_readers.text.plain import PlainTextReader
from xnmt.persistence import Serializable, serializable_init
from xnmt.vocabs import Vocab
from xnmt import output
from xnmt.sent import SegmentedSentence


class CharFromWordTextReader(PlainTextReader, Serializable):
  """
  Read in word based corpus and turned that into SegmentedSentence.
  SegmentedSentece's words are characters, but it contains the information of the segmentation.

  x = SegmentedSentence("i code today")
  (TRUE) x.words == ["i", "c", "o", "d", "e", "t", "o", "d", "a", "y"]
  (TRUE) x.segment == [0, 4, 9]

  It means that the segmentation (end of words) happen in the 0th, 4th and 9th position of the char sequence.
  """
  yaml_tag = "!CharFromWordTextReader"

  @serializable_init
  def __init__(self, vocab:Vocab=None, read_sent_len:bool=False, output_proc=None):
    self.vocab = vocab
    self.read_sent_len = read_sent_len
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

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
    sent_input = SegmentedSentence(segment=segs,
                                   words=[self.vocab.convert(c) for c in chars],
                                   idx=idx,
                                   vocab=self.vocab,
                                   output_procs=self.output_procs)
    return sent_input
