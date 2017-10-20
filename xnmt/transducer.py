class Transducer(object):
  """
  A transducer is a module that takes an object and transduces it into another object.
  
  Inputs and outputs can be DyNet expressions, ExpressionSequences, or anything else.
  
  The goal is to make features as modular as possible. It will be the responsibility of
  the user to ensure a transducer is being used in the correct context (i.e., supports
  correct input and output data types), therefore the expected data types should be made
  clear via appropriate argument naming and documentation.
  
  Transducers in general will have at least to methods:
  - __init__(...), should be used to configure the transducer. If possible, configuration
    should be transparent to a user and not require understanding of implementation
    details. If the transducer uses DyNet parameters, these must be initialized here.
  - __call__(...), will be perform the actual transduction and return the result 
  """
  def __call__(self, *args, **kwargs):
    """
    May take any parameters.
    :returns: result of transduction
    """
    raise NotImplementedError("subclasses must implement __call__()")

class SequenceTransducer(Transducer):
  """
  A special type of transducer that uses ExpressionSequence objects as inputs and outputs.
  ExpressionSequences are more powerful/flexible than plain DyNet expressions, so it is
  encouraged to implement sequence transducers whenever dealing with sequences. 
  """
  def __call__(self, *args, **kwargs):
    """
    Parameters should be ExpressionSequence objects
    :returns: result of transduction, a ExpressionSequence object
    """
    raise NotImplementedError("subclasses must implement __call__()")
  