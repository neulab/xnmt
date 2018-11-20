"""
A module defining triggers to the common events used throughout XNMT.
"""

from typing import Union
import numbers

from xnmt.train import tasks as training_tasks
from xnmt.models import base as models
from xnmt import batchers, events, losses, sent

@events.register_xnmt_event
def new_epoch(training_task: training_tasks.TrainingTask, num_sents: numbers.Integral) -> None:
  """
  Trigger event indicating a new epoch for the specified task.

  Args:
    training_task: task that proceeds into next epoch.
    num_sents: number of training sentences in new epoch.
  """
  pass

@events.register_xnmt_event
def set_train(val: bool) -> None:
  """
  Trigger event indicating enabling/disabling of "training" mode.

  Args:
    val: whether "training" mode is enabled
  """
  pass

@events.register_xnmt_event
def start_sent(src: Union[sent.Sentence, batchers.Batch]) -> None:
  """
  Trigger event indicating the start of a new sentence (or batch of sentences).

  Args:
    src: new sentence (or batch of sentences)
  """
  pass

@events.register_xnmt_event_sum
def calc_additional_loss(trg: Union[sent.Sentence, batchers.Batch],
                         parent_model: models.TrainableModel,
                         parent_model_loss: losses.FactoredLossExpr) -> losses.FactoredLossExpr:
  """
  Trigger event for calculating additional loss (e.g. reinforce loss) based on the reward

  Args:
    trg: Reference sentence
    parent_model: The reference to the parent model who called the addcitional_loss
    parent_model_loss: The loss from the parent_model.calc_loss()
  """
  return None

@events.register_xnmt_event_assign
def get_report_input(context: dict = {}) -> dict:
  return context

@events.register_xnmt_event
def set_reporting(reporting: bool) -> None:
  pass

