from typing import Any, List, Optional, Sequence
import numbers

from xnmt.persistence import serializable_init, Serializable

class Vocab(Serializable):
  """
  An open vocabulary that converts between strings and integer ids.

  The open vocabulary is realized via a special unknown-word token that is used whenever a word is not inside the
  list of known tokens.
  This class is immutable, i.e. its contents are not to change after the vocab has been initialized.

  For initialization, i2w or vocab_file must be specified, but not both.

  Args:
    i2w: complete list of known words, including ``<s>`` and ``</s>``.
    vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    sentencepiece_vocab: Set to ``True`` if ``vocab_file`` is the output of the sentencepiece tokenizer. Defaults to ``False``.
  """

  yaml_tag = "!Vocab"

  SS = 0
  ES = 1

  SS_STR = "<s>"
  ES_STR = "</s>"
  UNK_STR = "<unk>"

  @serializable_init
  def __init__(self,
               i2w: Optional[Sequence[str]] = None,
               vocab_file: Optional[str] = None,
               sentencepiece_vocab: bool = False) -> None:
    assert i2w is None or vocab_file is None
    assert i2w or vocab_file
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file, sentencepiece_vocab)
    assert i2w is not None
    self.i2w = i2w
    self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
    if Vocab.UNK_STR not in self.w2i:
      self.w2i[Vocab.UNK_STR] = len(self.i2w)
      self.i2w.append(Vocab.UNK_STR)
    self.unk_token = self.w2i[Vocab.UNK_STR]
    self.save_processed_arg("i2w", self.i2w)
    self.save_processed_arg("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file: str, sentencepiece_vocab: bool = False) -> List[str]:
    """Load the vocabulary from a file.
    
    If ``sentencepiece_vocab`` is set to True, this will accept a sentencepiece vocabulary file
    
    Args:
      vocab_file: file containing one word per line, and not containing ``<s>``, ``</s>``, ``<unk>``
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
    return self.w2i.get(w, self.unk_token)

  def __getitem__(self, i: numbers.Integral) -> str:
    return self.i2w[i]

  def __len__(self) -> int:
    return len(self.i2w)

  def is_compatible(self, other: Any) -> bool:
    """
    Check if this vocab produces the same conversions as another one.
    """
    if not isinstance(other, Vocab):
      return False
    if len(self) != len(other):
      return False
    if self.unk_token != other.unk_token:
      return False
    return self.w2i == other.w2i
