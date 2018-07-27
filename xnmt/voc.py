from xnmt.persistence import serializable_init, Serializable

class Vocab(Serializable):
  """
  Converts between strings and integer ids.

  Configured via either i2w or vocab_file (mutually exclusive).

  Args:
    i2w (list of string): list of words, including <s> and </s>
    vocab_file (str): file containing one word per line, and not containing <s>, </s>, <unk>
    sentencepiece_vocab (bool): Set to ``True`` if ``vocab_file`` is the output of the sentencepiece tokenizer. Defaults to ``False``.
  """

  yaml_tag = "!Vocab"

  SS = 0
  ES = 1

  SS_STR = "<s>"
  ES_STR = "</s>"
  UNK_STR = "<unk>"

  @serializable_init
  def __init__(self, i2w=None, vocab_file=None, sentencepiece_vocab=False):
    assert i2w is None or vocab_file is None
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file, sentencepiece_vocab)
    if i2w is not None:
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
    self.save_processed_arg("i2w", self.i2w)
    self.save_processed_arg("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file, sentencepiece_vocab=False):
    """Loads the vocabulary from a file.
    
    If ``sentencepiece_vocab`` is set to True, this will accept a sentencepiece vocabulary file
    
    Args:
      vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
      sentencepiece_vocab (bool): Set to ``True`` if ``vocab_file`` is the output of the sentencepiece tokenizer. Defaults to ``False``.
    """
    vocab = [Vocab.SS_STR, Vocab.ES_STR]
    reserved = {Vocab.SS_STR, Vocab.ES_STR, Vocab.UNK_STR}
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        # Sentencepiece vocab files have second field, ignore it
        if sentencepiece_vocab:
          word = word.split('\t')[0]
        if word in reserved:
          # Ignore if this is a sentencepiece vocab file
          if sentencepiece_vocab:
            continue
          else:
            raise RuntimeError(f"Vocab file {vocab_file} contains a reserved word: {word}")
        vocab.append(word)
    return vocab

  def convert(self, w: str) -> int:
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token is not None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i: int) -> str:
    return self.i2w[i]

  def __len__(self) -> int:
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

