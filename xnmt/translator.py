import dynet as dy

class Translator:
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''
  
  '''
  Calculate the loss of the input and output.
  '''
  def loss(self,x,y):
    raise NotImplementedError('loss must be implemented for Translator subclasses')

  '''
  Calculate the loss for a batch. By default, just iterate. Overload for better efficiency.
  '''
  def batch_loss(delf,xs,ys):
    return dy.esum( [self.loss(x,y) for x,y in zip(xs,ys)] )
