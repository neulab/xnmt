
class Evaluator:
  '''
  A class to evaluate the quality of output.
  '''
  
  '''
  Calculate the quality of output given a references.
  '''
  def evaluate(self, ref, hyp):
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')
