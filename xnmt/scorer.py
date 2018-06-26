import dynet as dy
from typing import List, Union, Optional

from xnmt.param_init import ParamInitializer, GlorotInitializer, ZeroInitializer
from xnmt.param_collection import ParamManager
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.transform import Linear
from xnmt import batcher, vocab, input_reader

class Scorer(object):
  """
  A template class of things that take in a vector and produce a
  score over discrete output items.
  """

  def calc_scores(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the score of each discrete decision, where the higher
    the score is the better the model thinks a decision is. These
    often correspond to unnormalized log probabilities.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_score must be implemented by subclasses of Scorer')

  def calc_probs(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the normalized probability of a decision.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_prob must be implemented by subclasses of Scorer')

  def calc_log_probs(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the log probability of a decision
    
    log(calc_prob()) == calc_log_prob()

    Both functions exist because it might help save memory.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_log_prob must be implemented by subclasses of Scorer')

  def calc_loss(self, x: dy.Expression, y: Union[int, List[int]]) -> dy.Expression:
    """
    Calculate the loss incurred by making a particular decision.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_loss must be implemented by subclasses of Scorer')

  def _choose_vocab_size(self, vocab_size: Optional[int], vocab: Optional[vocab.Vocab],
                         trg_reader: Optional[input_reader.InputReader]) -> int:
    """Choose the vocab size for the embedder based on the passed arguments.

    This is done in order of priority of vocab_size, vocab, model

    Args:
      vocab_size: vocab size or None
      vocab: vocab or None
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab size
    """
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif trg_reader is None or trg_reader.vocab is None:
      raise ValueError(
        "Could not determine scorer's's output size. "
        "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)

class Softmax(Scorer, Serializable):
  """
  A class that does an affine transform from the input to the vocabulary size,
  and calculates a softmax.

  Note that all functions in this class rely on calc_scores(), and thus this
  class can be sub-classed by any other class that has an alternative method
  for calculating un-normalized log probabilities by simply overloading the
  calc_scores() function.

  Args:
    input_dim: Size of the input vector
    vocab_size: Size of the vocab to predict
    vocab: A vocab object from which the vocab size can be derived automatically
    trg_reader: An input reader for the target, which can be used to derive the vocab size
    label_smoothing: Whether to apply label smoothing (a value of 0.1 is good if so)
    param_init: How to initialize the parameters
    bias_init: How to initialize the bias
    output_projector: The projection to be used before the output
  """

  yaml_tag = '!Softmax'

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),               
               vocab_size: Optional[int] = None,
               vocab: Optional[vocab.Vocab] = None,
               trg_reader: Optional[input_reader.InputReader] = Ref("model.trg_reader", default=None),
               label_smoothing: float = 0.0,
               param_init: ParamInitializer = Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init: ParamInitializer = Ref("exp_global.bias_init", default=bare(ZeroInitializer)),
               output_projector: Linear = None) -> None:
    self.param_col = ParamManager.my_params(self)
    self.input_dim = input_dim
    self.output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.label_smoothing = label_smoothing

    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or Linear(
                                                              input_dim=self.input_dim, output_dim=self.output_dim,
                                                              param_init=param_init, bias_init=bias_init))
  
  def calc_scores(self, x: dy.Expression) -> dy.Expression:
    return self.output_projector(x)

  def calc_loss(self, x: dy.Expression, y: Union[int, List[int]]) -> dy.Expression:

    scores = self.calc_scores(x)

    if self.label_smoothing == 0.0:
      # single mode
      if not batcher.is_batched(y):
        loss = dy.pickneglogsoftmax(scores, y)
      # minibatch mode
      else:
        loss = dy.pickneglogsoftmax_batch(scores, y)
    else:
      log_prob = dy.log_softmax(scores)
      if not batcher.is_batched(y):
        pre_loss = -dy.pick(log_prob, y)
      else:
        pre_loss = -dy.pick_batch(log_prob, y)

      ls_loss = -dy.mean_elems(log_prob)
      loss = ((1 - self.label_smoothing) * pre_loss) + (self.label_smoothing * ls_loss)
    
    return loss

  def calc_probs(self, x: dy.Expression) -> dy.Expression:
    return dy.softmax(self.calc_scores(x))

  def calc_log_probs(self, x: dy.Expression) -> dy.Expression:
    return dy.log_softmax(self.calc_scores(x))

