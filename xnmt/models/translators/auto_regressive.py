import dynet as dy
import collections

from typing import Optional, Sequence, Union

import xnmt.batchers as batchers
import xnmt.sent as sent
import xnmt.vocabs as vocabs
import xnmt.models.base as base


class AutoRegressiveTranslator(base.ConditionedModel, base.GeneratorModel):
  """
  A template class for auto-regressive translators.
  The core methods are calc_nll, add_input, best_k, and sample.
  The first is used during training, the latter three for inference.
  Similarly during inference, a search strategy is used to generate an output sequence by repeatedly calling
  add_input and either best_k or sample.
  """

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) -> dy.Expression:
    """
    Calculate the negative log likelihood, or similar value, of trg given src
    Args:
      src: The input
      trg: The output
    Return:
      The likelihood
    """
    raise NotImplementedError('must be implemented by subclasses')

  def generate(self,
               src: batchers.Batch,
               search_strategy: 'search_strategies.SearchStrategy') -> Sequence[sent.Sentence]:
    raise NotImplementedError("must be implemented by subclasses")

  def set_trg_vocab(self, trg_vocab: Optional[vocabs.Vocab] = None) -> None:
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.
    Args:
      trg_vocab: target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  # The output template for the auto regressive translator output
  Output = collections.namedtuple('Output', ['dec_state', 'att_state', 'attention'])
