import dynet as dy

class Attender:
  '''
  A template class for functions implementing attention.
  '''
  
  '''
  Implement things.
  '''
  def loss(self,x,y):
    raise NotImplementedError('loss must be implemented for Translator subclasses')

  '''
  Return an 
  '''
  def batch_loss(delf,xs,ys):
    return dy.esum( [self.loss(x,y) for x,y in zip(xs,ys)] )
