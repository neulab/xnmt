import dynet as dy

from typing import Any, Sequence, Union

from xnmt import sent, batchers
from xnmt.models import ConditionedModel, GeneratorModel
from xnmt.models.translators import TranslatorOutput
from xnmt.modelparts.decoders import AutoRegressiveDecoderState


class AutoRegressiveTranslator(ConditionedModel, GeneratorModel):
  """
  A template class for auto-regressive translators.
  The core methods are calc_nll and generate / generate_one_step.
  The former is used during training, the latter for inference.
  Similarly during inference, a search strategy is used to generate an output sequence by repeatedly calling
  generate_one_step.
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

  def generate(self, src, search_strategy, forced_trg_ids=None) -> Sequence[sent.Sentence]:
    raise NotImplementedError("must be implemented by subclasses")

  def generate_one_step(self, current_word: Any, current_state: AutoRegressiveDecoderState) -> TranslatorOutput:
    raise NotImplementedError("must be implemented by subclasses")

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.
    Args:
      trg_vocab (Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def get_nobp_state(self, state):
    return dy.nobackprop(state.rnn_state.output())

