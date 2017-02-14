
class Output:
  '''
  A template class to represent all output.
  '''

  def __init__(self):
    ''' Initialize an empty output. '''
    self.actions = []

  def __init__(self, actions):
    ''' Initialize an output with actions. '''
    self.actions = actions

  def to_string(self):
    raise NotImplementedError('All outputs must implement to_string.')
