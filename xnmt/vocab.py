import io
from xnmt.serialize.serializable import Serializable

class Vocab(Serializable):
  '''
  Converts between strings and integer ids
  '''

  yaml_tag = "!Vocab"

  SS = 0
  ES = 1

  SS_STR = u"<s>"
  ES_STR = u"</s>"
  UNK_STR = u"<unk>"

  def __init__(self, i2w=None, vocab_file=None):
    """
    :param i2w: list of words, including <s> and </s>
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    i2w and vocab_file are mutually exclusive
    """
    if not (i2w is None or vocab_file is None):
      print("break")
    assert i2w is None or vocab_file is None
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file)
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
      self.frozen = True
    else :
      self.w2i = {}
      self.i2w = []
      self.unk_token = None
      self.w2i[self.SS_STR] = self.SS
      self.w2i[self.ES_STR] = self.ES
      self.i2w.append(self.SS_STR)
      self.i2w.append(self.ES_STR)
      self.frozen = False
    self.overwrite_serialize_param("i2w", self.i2w)
    self.overwrite_serialize_param("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file):
    """
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = [Vocab.SS_STR, Vocab.ES_STR]
    reserved = set([Vocab.SS_STR, Vocab.ES_STR, Vocab.UNK_STR])
    with io.open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        if word in reserved:
          raise RuntimeError(f"Vocab file {vocab_file} contains a reserved word: {word}")
        vocab.append(word)
    return vocab

  def convert(self, w):
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token != None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

  def freeze(self):
    self.frozen = True

  def set_unk(self, w):
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]
