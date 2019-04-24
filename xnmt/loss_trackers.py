from typing import Optional, Union
import time
import numbers

from xnmt import batchers, events, logger, losses, sent, utils
from xnmt.eval import metrics
from xnmt.persistence import Ref, serializable_init, Serializable

class AccumTimeTracker(object):
  def __init__(self) -> None:
    self.start_time = None
    self.accum_time = 0.0

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, *args):
    self.accum_time += time.time() - self.start_time

  def get_and_reset(self) -> numbers.Real:
    ret = self.accum_time
    self.accum_time = 0.0
    return ret

class TrainLossTracker(Serializable):

  """
  Loss tracker for the loss of a training task.

  Args:
    accumulative: whether to accumulate (average) training loss over each current epoch
    aux_loss_per_token: whether to normalize auxiliary losses by number of tokens
  """

  yaml_tag = "!TrainLossTracker"

  REPORT_TEMPLATE_SPEED = 'Epoch {epoch:.4f}: {data_name}_loss/word={loss:.6f} (steps={n_iter}, words/sec={words_per_sec:.2f}, time={time})'
  REPORT_TEMPLATE       = 'Epoch {epoch:.4f}: {data_name}_loss/word={loss:.6f} (steps={n_iter}, words/sec={words_per_sec}, time={time})'
  REPORT_TEMPLATE_ADDITIONAL = '- {loss_name} {loss:5.6f}'
  REPORT_EVERY = 1000

  @serializable_init
  @events.register_xnmt_handler
  def __init__(self, accumulative: bool = False, aux_loss_per_token = True) -> None:
    self.accumulative = accumulative
    self.training_task = None
    self.epoch_loss = losses.FactoredLossVal()
    self.epoch_words = 0
    self.last_report_sents_into_epoch = 0
    self.last_report_sents_since_start = 0

    self.last_report_words = 0

    self.time_tracker = AccumTimeTracker()
    self.start_time = time.time()
    self.aux_loss_per_token = aux_loss_per_token

  def set_training_task(self, training_task: 'xnmt.train.tasks.TrainingTask') -> None:
    self.training_task = training_task
    self.name = self.training_task.name

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    if training_task is self.training_task:
      self.epoch_loss.clear()
      self.epoch_words = 0
      self.last_report_sents_since_start = 0
      self.last_report_words = 0
      self.loss_normalizer_tok = 0
      self.loss_normalizer_sent = 0

  def report(self, trg: Union[sent.Sequence, batchers.Batch], loss: losses.FactoredLossVal) -> None:
    """
    Accumulate training loss and report every REPORT_EVERY sentences.
    """
    self.epoch_words += self.count_trg_words(trg)
    self.epoch_loss += loss
    self.loss_normalizer_sent += trg.batch_size() if batchers.is_batched(trg) else 1
    if self.accumulative:
      self.loss_normalizer_tok = self.epoch_words
    else:
      self.loss_normalizer_tok += self.count_trg_words(trg)


    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    should_report = sent_num_not_report >= TrainLossTracker.REPORT_EVERY \
                    or self.training_task.training_state.sents_into_epoch == self.training_task.cur_num_sentences()

    if should_report:
      fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                         + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
      accum_time = self.time_tracker.get_and_reset()
      rep_train_loss = self.epoch_loss.sum_factors() / self.loss_normalizer_tok
      utils.log_readable_and_tensorboard(
        template = TrainLossTracker.REPORT_TEMPLATE_SPEED if accum_time else TrainLossTracker.REPORT_TEMPLATE,
        args = {"loss": rep_train_loss},
        n_iter = self.training_task.training_state.steps_since_start,
        fractional_epoch = fractional_epoch,
        time = utils.format_time(time.time() - self.start_time),
        words = self.epoch_words,
        data_name = "train",
        task_name = self.name,
        words_per_sec = (self.epoch_words - self.last_report_words) / accum_time if accum_time else None
      )

      if len(self.epoch_loss) > 1:
        for loss_i, (loss_name, loss_values) in enumerate(self.epoch_loss.items()):
          cur_normalizer = (self.loss_normalizer_tok if loss_i==0 or self.aux_loss_per_token else self.loss_normalizer_sent)
          utils.log_readable_and_tensorboard(template=TrainLossTracker.REPORT_TEMPLATE_ADDITIONAL,
                                             args={loss_name: loss_values / cur_normalizer},
                                             n_iter=self.training_task.training_state.steps_since_start,
                                             fractional_epoch=fractional_epoch,
                                             data_name="train",
                                             task_name=self.name,
                                             loss_name=loss_name,
                                             loss=loss_values / cur_normalizer,
                                             )

      self.last_report_words = self.epoch_words
      self.last_report_sents_since_start = self.training_task.training_state.sents_since_start
      if not self.accumulative:
        self.epoch_loss.clear()
        self.loss_normalizer_sent = 0
        self.loss_normalizer_tok = 0

  def count_trg_words(self, trg_words: Union[sent.Sentence, batchers.Batch]) -> int:
    if isinstance(trg_words, batchers.Batch):
      return sum(inp.len_unpadded() for inp in trg_words)
    else:
      return trg_words.len_unpadded()

class DevLossTracker(TrainLossTracker, Serializable):

  yaml_tag = "!DevLossTracker"

  REPORT_TEMPLATE_DEV         = 'Epoch {epoch:.4f} dev {score} (time={time})'
  REPORT_TEMPLATE_DEV_AUX     = '             dev auxiliary {score}'
  REPORT_TEMPLATE_TIME_NEEDED = '             checkpoint took {time_needed}'

  """
  Loss tracker for dev checkpoints.
  
  Args:
    dev_every: dev checkpoints every n sentences (0 for only after epoch)
  """

  @serializable_init
  def __init__(self,
               eval_every: numbers.Integral = 0,
               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum")) -> None:
    self.loss_comb_method = loss_comb_method
    self.training_task = None
    self.eval_dev_every = eval_every

    self.last_report_sents_since_start = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.aux_scores = []

    self.start_time = time.time()
    self.time_tracker = AccumTimeTracker()

  def set_dev_score(self, dev_score: metrics.EvalScore) -> None:
    self.dev_score = dev_score

  def add_aux_score(self, score: metrics.EvalScore) -> None:
    self.aux_scores.append(score)

  def should_report_dev(self) -> bool:
    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    if self.eval_dev_every > 0:
      return sent_num_not_report >= self.eval_dev_every
    else:
      return sent_num_not_report >= self.training_task.cur_num_sentences()

  def report(self) -> None:
    this_report_time = time.time()
    self.last_report_sents_since_start = self.training_task.training_state.sents_since_start
    self.fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                            + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
    dev_time = self.time_tracker.get_and_reset()
    utils.log_readable_and_tensorboard(template=DevLossTracker.REPORT_TEMPLATE_DEV,
                                       args={self.dev_score.metric_name(): self.dev_score.value()},
                                       n_iter=self.training_task.training_state.steps_since_start,
                                       fractional_epoch=self.fractional_epoch,
                                       data_name="dev",
                                       task_name=self.name,
                                       score=self.dev_score,
                                       time=utils.format_time(this_report_time - self.start_time))
    for score in self.aux_scores:
      utils.log_readable_and_tensorboard(template=DevLossTracker.REPORT_TEMPLATE_DEV_AUX,
                                         args={score.metric_name(): score.value()},
                                         n_iter=self.training_task.training_state.steps_since_start,
                                         fractional_epoch=self.fractional_epoch,
                                         data_name="dev",
                                         task_name=self.name,
                                         score=score)
    logger.info(DevLossTracker.REPORT_TEMPLATE_TIME_NEEDED.format(time_needed= utils.format_time(dev_time),
                                                                  extra={"task_name" : self.name}))
    self.aux_scores = []

