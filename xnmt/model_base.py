from typing import Optional, Sequence, Union

from xnmt import batcher, events, input_reader, loss, output, training_task
import xnmt.loss_calculator
import xnmt.input
from xnmt.persistence import Serializable, serializable_init

class TrainableModel(object):
  """
  A template class for a basic trainable model, implementing a loss function.
  """

  def calc_loss(self, *args, **kwargs) -> loss.FactoredLossExpr:
    """Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Arguments are to be defined by subclasses

    Returns:
      A (possibly batched) expression representing the loss.
    """

  def get_primary_loss(self) -> str:
    """
    Returns:
      Identifier for primary loss.
    """
    raise NotImplementedError("Pick a key for primary loss that is used for dev_loss calculation")

class UnconditionedModel(TrainableModel):
  """
  A template class for trainable model that computes target losses without conditioning on other inputs.

  Args:
    trg_reader: target reader
  """

  def __init__(self, trg_reader: input_reader.InputReader):
    self.trg_reader = trg_reader

  def calc_loss(self, trg: Union[batcher.Batch, xnmt.input.Input]) -> loss.FactoredLossExpr:
    """Calculate loss based on target inputs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      trg: The target, a sentence or a batch of sentences.

    Returns:
      A (possibly batched) expression representing the loss.
    """


class ConditionedModel(TrainableModel):
  """
  A template class for a trainable model that computes target losses conditioned on a source input.

  Args:
    src_reader: source reader
    trg_reader: target reader
  """

  def __init__(self, src_reader: input_reader.InputReader, trg_reader: input_reader.InputReader):
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def calc_loss(self, src: Union[batcher.Batch, xnmt.input.Input], trg: Union[batcher.Batch, xnmt.input.Input],
                loss_calculator: xnmt.loss_calculator.LossCalculator) -> loss.FactoredLossExpr:
    """Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      src: The source, a sentence or a batch of sentences.
      trg: The target, a sentence or a batch of sentences.
      loss_calculator: loss calculator.

    Returns:
      A (possibly batched) expression representing the loss.
    """


class GeneratorModel(object):
  """
  A template class for models that can perform inference to generate some kind of output.

  Args:
    src_reader: source input reader
    trg_reader: an optional target input reader, needed in some cases such as n-best scoring
  """
  def __init__(self, src_reader: input_reader.InputReader, trg_reader: Optional[input_reader.InputReader] = None) \
          -> None:
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def generate(self, src: batcher.Batch, idx: Sequence[int], *args, **kwargs) -> Sequence[output.Output]:
    """
    Generate outputs.

    Args:
      src: batch of source-side inputs (:class:``xnmt.input.Input``)
      idx: list of integers specifying the place of the input sentences in the test corpus
      *args:
      **kwargs: Further arguments to be specified by subclasses
    Returns:
      output objects
    """
    raise NotImplementedError("must be implemented by subclasses")

class EventTrigger(object):
  """
  A template class defining triggers to the common events used throughout XNMT.
  """
  @events.register_xnmt_event
  def new_epoch(self, training_task: training_task.TrainingTask, num_sents: int) -> None:
    """
    Trigger event indicating a new epoch for the specified task.

    Args:
      training_task: task that proceeds into next epoch.
      num_sents: number of training sentences in new epoch.
    """
    pass

  @events.register_xnmt_event
  def set_train(self, val: bool) -> None:
    """
    Trigger event indicating enabling/disabling of "training" mode.

    Args:
      val: whether "training" mode is enabled
    """
    pass

  @events.register_xnmt_event
  def start_sent(self, src: Union[xnmt.input.Input, batcher.Batch]) -> None:
    """
    Trigger event indicating the start of a new sentence (or batch of sentences).

    Args:
      src: new sentence (or batch of sentences)
    """
    pass

  @events.register_xnmt_event_sum
  def calc_additional_loss(self,
                           trg: Union[xnmt.input.Input, batcher.Batch],
                           parent_model: TrainableModel,
                           parent_model_loss: loss.FactoredLossExpr) -> loss.FactoredLossExpr:
    """
    Trigger event for calculating additional loss (e.g. reinforce loss) based on the reward

    Args:
      trg: Reference sentence
      parent_model: The reference to the parent model who called the addcitional_loss
      parent_model_loss: The loss from the parent_model.calc_loss()
    """
    return None

class CascadeGenerator(GeneratorModel, EventTrigger, Serializable):
  """
  A cascade that chains several generator models.

  This generator does not support calling ``generate()`` directly. Instead, it's sub-generators should be accessed
  and used to generate outputs one by one.

  Args:
    generators: list of generators
  """
  yaml_tag = '!CascadeGenerator'

  @serializable_init
  def __init__(self, generators: Sequence[GeneratorModel]) -> None:
    super().__init__(src_reader = generators[0].src_reader, trg_reader = generators[-1].trg_reader)
    self.generators = generators

  def generate(self, *args, **kwargs):
    raise ValueError("cannot call CascadeGenerator.generate() directly; access the sub-generators instead.")
