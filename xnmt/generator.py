from xnmt.events import register_xnmt_event, register_xnmt_event_sum
from xnmt.report import Reportable

class GeneratorModel(Reportable):
  """
  A model that generates output from a given input.
  """
  def generate_output(self, *args, **kwargs):
    output = self.generate(*args, **kwargs)
    if hasattr(self, "post_processor"):
      self.post_processor.process_outputs(output)
    return output

  def generate(self, *args, **kwargs):
    raise NotImplementedError()

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

  @register_xnmt_event
  def new_epoch(self, training_task, num_sents):
    pass

  @register_xnmt_event
  def set_train(self, val):
    pass

  @register_xnmt_event
  def start_sent(self, src):
    pass

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None):
    raise NotImplementedError()

  def get_primary_loss(self):
    raise NotImplementedError("Pick a key for primary loss that is used for dev_loss calculation")

  @register_xnmt_event_sum
  def calc_additional_loss(self, src, trg, translator_loss, trg_counts):
    return None


