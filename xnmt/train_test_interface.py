
class TrainTestInterface(object):
  """
  All subcomponents of the translator that behave differently at train and test time
  should subclass this class.
  """
  def set_train(self, val):
    """
    Will be called with val=True when starting to train, and with val=False when starting
    to evaluate.
    :param val: bool that indicates whether we're in training mode
    """
    pass

  def get_train_test_components(self):
    """
    :returns: list of subcomponents that implement TrainTestInterface and will be called recursively.
    """
    return []

  def receive_decoder_loss(self, loss):
    return None
