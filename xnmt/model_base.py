from typing import Union

from xnmt import batcher, events, input, input_reader, loss, loss_calculator, output, training_task

class TrainableModel(object):
  """
  A template class for a basic trainable model, implementing a loss function based on inputs and outputs.

  Args:
    src_reader: source reader
    trg_reader: target reader
  """

  def __init__(self, src_reader: input_reader.InputReader, trg_reader: input_reader.InputReader):
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def calc_loss(self, src: Union[batcher.Batch, input.Input], trg: Union[batcher.Batch, input.Input],
                loss_calculator: loss_calculator.LossCalculator) -> loss.FactoredLossExpr:
    '''Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      src: The source, a sentence or a batch of sentences.
      trg: The target, a sentence or a batch of sentences.
      loss_calculator: loss calculator.

    Returns:
      A (possibly batched) expression representing the loss.
    '''

  def get_primary_loss(self) -> str:
    """
    Returns:
      Identifier for primary loss.
    """
    raise NotImplementedError("Pick a key for primary loss that is used for dev_loss calculation")


class GeneratorModel(TrainableModel):
  """
  A template class for trainable models that can perform inference to generate some kind of output.
  """

  def initialize_generator(self, **kwargs):
    """
    Initialize generator.
    """
    pass

  def generate_output(self, *args, **kwargs) -> output.Output:
    """
    Generate post-processed outputs.
    """
    generation_output = self.generate(*args, **kwargs)
    if hasattr(self, "post_processor"):
      self.post_processor.process_outputs(generation_output)
    return generation_output

  def generate(self, *args, **kwargs) -> output.Output:
    """
    Generate unprocessed outputs.
    """
    raise NotImplementedError()


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
  def start_sent(self, src: Union[input.Input, batcher.Batch]) -> None:
    """
    Trigger event indicating the start of a new sentence (or batch of sentences).

    Args:
      src: new sentence (or batch of sentences)
    """
    pass

  @events.register_xnmt_event_sum
  def calc_additional_loss(self, reward):
    """
    Trigger event for calculating additional loss (e.g. reinforce loss) based on the reward

    Args:
      reward: The default is log likelihood (-1 * calc_loss).
    """
    return None
