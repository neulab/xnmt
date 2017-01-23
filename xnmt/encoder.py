
class Encoder:
  '''
  A template class to encode an input.
  '''
  
  '''
  Takes an Input and returns an EncodedInput.
  '''
  def encode(self,x):
    raise NotImplementedError('encode must be implemented in Encoder subclasses')
