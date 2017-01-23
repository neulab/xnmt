
class Embedder:
  '''
  A template class to embed a word or token.
  '''
  
  '''
  Takes a string or word ID and returns its embedding.
  '''
  def embed(self,x):
    raise NotImplementedError('embed must be implemented in Embedder subclasses')
