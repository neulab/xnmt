from xnmt.persistence import serializable_init, Serializable

SS = 0
ES = 1

SS_STR = "<s>"
ES_STR = "</s>"
UNK_STR = "<unk>"

class Vocab(Serializable):
  '''
  Converts between strings and integer ids.
  
  Configured via either i2w or vocab_file (mutually exclusive).
  
  Args:
    i2w (list of string): list of words, including <s> and </s>
    vocab_file (str): file containing one word per line, and not containing <s>, </s>, <unk>
  '''

  yaml_tag = "!Vocab"

  @serializable_init
  def __init__(self, i2w=None, vocab_file=None):
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
      self.w2i[SS_STR] = SS
      self.w2i[ES_STR] = ES
      self.i2w.append(SS_STR)
      self.i2w.append(ES_STR)
      self.frozen = False
    self.save_processed_arg("i2w", self.i2w)
    self.save_processed_arg("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file):
    """
    Args:
      vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = [SS_STR, ES_STR]
    reserved = set([SS_STR, ES_STR, UNK_STR])
    with open(vocab_file, encoding='utf-8') as f:
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

  def is_compatible(self, other):
    """
    Check if this vocab produces the same conversions as another one.
    """
    if not isinstance(other, Vocab):
      return False
    if len(self) != len(other):
      return False
    if self.frozen != other.frozen or self.unk_token != other.unk_token:
      return False
    return self.w2i == other.w2i

  def freeze(self):
    """
    Mark this vocab as fixed, so no further words can be added. Only after freezing can the unknown word token be set.
    """
    self.frozen = True

  def set_unk(self, w):
    """
    Sets the unknown word token. Can only be invoked after calling freeze().
    
    Args:
      w (str): unknown word token
    """
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]
