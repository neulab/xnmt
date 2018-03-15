from xnmt.events import register_xnmt_event, register_xnmt_event_sum

class GeneratorModel(object):
  # TODO: document me
  def generate_output(self, *args, **kwargs):
    # Generate the output
    generation_output = self.generate(*args, **kwargs)
    # Post process it
    if hasattr(self, "post_processor"):
      self.post_processor.process_outputs(generation_output)
    return generation_output

  def generate(self, *args, **kwargs):
    raise NotImplementedError()

  @register_xnmt_event
  def initialize_generator(self, **kwargs):
    pass

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
  def calc_additional_loss(self, reward):
    ''' Calculate reinforce loss based on the reward
    Args:
      reward: The default is log likelihood (-1 * calc_loss).
    '''
    return None

