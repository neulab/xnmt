import numpy as np

from typing import Optional

from xnmt import output
from xnmt.vocabs import Vocab
from xnmt.sent import SimpleSentence
from xnmt.input_readers.text.base import BaseTextReader
from xnmt.persistence import Serializable, serializable_init
from xnmt.events import register_xnmt_handler, handle_xnmt_event


class RamlTextReader(BaseTextReader, Serializable):
  """
  Handles the RAML sampling, can be used on the target side, or on both the source and target side.
  Randomly replaces words according to Hamming Distance.

  https://arxiv.org/pdf/1808.07512.pdf
  https://arxiv.org/pdf/1609.00150.pdf
  """
  yaml_tag = '!RamlTextReader'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, tau: Optional[float] = 1., vocab: Optional[Vocab] = None, output_proc=None):
    """
    Args:
      tau: The temperature that controls peakiness of the sampling distribution
      vocab: The vocabulary
    """
    self.tau = tau
    self.vocab = vocab
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)
    self.train = False

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def read_sent(self, line, idx):
    words = line.strip().split()
    if not self.train:
      return SimpleSentence(idx=idx,
                            words=[self.vocab.convert(word) for word in words] + [Vocab.ES],
                            vocab=self.vocab,
                            output_procs=self.output_procs)
    word_ids = np.array([self.vocab.convert(word) for word in words])
    length = len(word_ids)
    logits = np.arange(length) * (-1) * self.tau
    logits = np.exp(logits - np.max(logits))
    probs = logits / np.sum(logits)
    num_words = np.random.choice(length, p=probs)
    corrupt_pos = np.random.binomial(1, p=num_words/length, size=(length,))
    num_words_to_sample = np.sum(corrupt_pos)
    sampled_words = np.random.choice(np.arange(2, len(self.vocab)), size=(num_words_to_sample,))
    word_ids[np.where(corrupt_pos==1)[0].tolist()] = sampled_words
    return SimpleSentence(idx=idx,
                          words=word_ids.tolist() + [Vocab.ES],
                          vocab=self.vocab,
                          output_procs=self.output_procs)

  def needs_reload(self) -> bool:
    # TODO(philip30): Does it need really a reload?
    return True
