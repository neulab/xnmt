from typing import List, Union, Optional
import numbers

import dynet as dy

from xnmt import batchers, input_readers, param_collections, param_initializers, vocabs, logger
from xnmt.modelparts import transforms
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.events import handle_xnmt_event, register_xnmt_handler

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
      y: The correct label(s)
    """
    raise NotImplementedError('calc_loss must be implemented by subclasses of Scorer')

  def _choose_vocab_size(self, vocab_size: Optional[int], vocab: Optional[vocabs.Vocab],
                         trg_reader: Optional[input_readers.InputReader]) -> int:
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
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               vocab_size: Optional[numbers.Integral] = None,
               vocab: Optional[vocabs.Vocab] = None,
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None),
               label_smoothing: numbers.Real = 0.0,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               output_projector: transforms.Linear = None) -> None:
    self.param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.label_smoothing = label_smoothing

    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or transforms.Linear(
                                                              input_dim=self.input_dim, output_dim=self.output_dim,
                                                              param_init=param_init, bias_init=bias_init))
  
  def calc_scores(self, x: dy.Expression) -> dy.Expression:
    return self.output_projector.transform(x)
  
  def can_loss_be_derived_from_scores(self):
    """
    This method can be used to determine whether dy.pickneglogsoftmax can be used to fastly calculate the loss value.
    If False, then the calc_loss method should calc (1) log_softmax, (2) perform necessary modification, (3) pick the loss
    """
    return self.label_smoothing == 0.0

  def calc_loss(self, x: dy.Expression, y: Union[numbers.Integral, List[numbers.Integral]]) -> dy.Expression:
    if self.can_loss_be_derived_from_scores():
      scores = self.calc_scores(x)
      # single mode
      if not batchers.is_batched(y):
        loss = dy.pickneglogsoftmax(scores, y)
      # minibatch mode
      else:
        loss = dy.pickneglogsoftmax_batch(scores, y)
    else:
      log_prob = self.calc_log_probs(x)
      if not batchers.is_batched(y):
        loss = -dy.pick(log_prob, y)
      else:
        loss = -dy.pick_batch(log_prob, y)

      if self.label_smoothing > 0:
        ls_loss = -dy.mean_elems(log_prob)
        loss = ((1 - self.label_smoothing) * loss) + (self.label_smoothing * ls_loss)
    
    return loss

  def calc_probs(self, x: dy.Expression) -> dy.Expression:
    return dy.softmax(self.calc_scores(x))

  def calc_log_probs(self, x: dy.Expression) -> dy.Expression:
    return dy.log_softmax(self.calc_scores(x))


class LexiconSoftmax(Softmax, Serializable):
  """
    A subclass of the softmax class that can make use of an external lexicon probability as described in:
    http://anthology.aclweb.org/D/D16/D16-1162.pdf

    Args:
      input_dim: Size of the input vector
      vocab_size: Size of the vocab to predict
      vocab: A vocab object from which the vocab size can be derived automatically
      trg_reader: An input reader for the target, which can be used to derive the vocab size
      label_smoothing: Whether to apply label smoothing (a value of 0.1 is good if so)
      param_init: How to initialize the parameters
      bias_init: How to initialize the bias
      output_projector: The projection to be used before the output
      lexicon_file: A file containing "trg src p(trg|src)"
      lexicon_alpha: smoothing constant for bias method
      lexicon_type: Either bias or linear method
    """
  
  yaml_tag = '!LexiconSoftmax'

  @serializable_init
  @register_xnmt_handler
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               vocab_size: Optional[numbers.Integral] = None,
               vocab: Optional[vocabs.Vocab] = None,
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None),
               attender = Ref("model.attender"),
               label_smoothing: numbers.Real = 0.0,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init",
                                                                    default=bare(param_initializers.ZeroInitializer)),
               output_projector: transforms.Linear = None,
               lexicon_file=None,
               lexicon_alpha=0.001,
               lexicon_type='bias',
               coef_predictor: transforms.Linear = None,
               src_vocab = Ref("model.src_reader.vocab", default=None)) -> None:
    self.param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    self.output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.label_smoothing = label_smoothing
  
    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or transforms.Linear(
                                                              input_dim=self.input_dim, output_dim=self.output_dim,
                                                              param_init=param_init, bias_init=bias_init))
    self.coef_predictor = self.add_serializable_component("coef_predictor", coef_predictor,
                                                          lambda: coef_predictor or transforms.Linear(
                                                            input_dim=self.input_dim, output_dim=1,
                                                            param_init=param_init, bias_init=bias_init
                                                          ))
    self.lexicon_file = lexicon_file
    self.lexicon_type = lexicon_type
    self.lexicon_alpha = lexicon_alpha
    
    assert lexicon_type in ["bias", "linear"], "Lexicon type can be either 'bias' or 'linear' only!"
    # Reference to other parts of the model
    self.src_vocab = src_vocab
    self.trg_vocab = vocab if vocab is not None else trg_reader.vocab
    self.attender = attender
    # Sparse data structure to store exteranl lexicon prob
    self.lexicon = None
    # State of the sofmax
    self.lexicon_prob = None
    self.coeff = None
    self.dict_prob = None
    
  def load_lexicon(self):
    logger.info("Loading lexicon from file: " + self.lexicon_file)
    lexicon = [{} for _ in range(len(self.src_vocab))]
    with open(self.lexicon_file, encoding='utf-8') as fp:
      for line in fp:
        try:
          trg, src, prob = line.rstrip().split()
        except:
          logger.warning("Failed to parse 'trg src prob' from:" + line.strip())
          continue
        trg_id = self.trg_vocab.convert(trg)
        src_id = self.src_vocab.convert(src)
        lexicon[src_id][trg_id] = float(prob)
    # Setting the rest of the weight to the unknown word
    for i in range(len(lexicon)):
      sum_prob = sum(lexicon[i].values())
      if sum_prob < 1.0:
        lexicon[i][self.trg_vocab.convert(self.trg_vocab.unk_token)] = 1.0 - sum_prob
    # Overriding special tokens
    src_unk_id = self.src_vocab.convert(self.src_vocab.unk_token)
    trg_unk_id = self.trg_vocab.convert(self.trg_vocab.unk_token)
    lexicon[self.src_vocab.SS] = {self.trg_vocab.SS: 1.0}
    lexicon[self.src_vocab.ES] = {self.trg_vocab.ES: 1.0}
    # TODO(philip30): Note sure if this is intended
    lexicon[src_unk_id] = {trg_unk_id: 1.0}
    return lexicon

  @handle_xnmt_event
  def on_new_epoch(self, *args, **kwargs):
    if self.lexicon is None:
      self.lexicon = self.load_lexicon()
  
  @handle_xnmt_event
  def on_start_sent(self, src):
    self.coeff = None
    self.dict_prob = None
    
    batch_size = src.batch_size()
    col_size = src.sent_len()
  
    idxs = [(x, j, i) for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].keys()]
    idxs = tuple(map(list, list(zip(*idxs))))
  
    values = [x for i in range(batch_size) for j in range(col_size) for x in self.lexicon[src[i][j]].values()]
    dim = len(self.trg_vocab), col_size, batch_size
    self.lexicon_prob = dy.nobackprop(dy.sparse_inputTensor(idxs, values, dim, batched=True))
  
  def calc_scores(self, x: dy.Expression) -> dy.Expression:
    model_score = self.output_projector.transform(x)
    if self.lexicon_type == 'bias':
      model_score += dy.sum_dim(dy.log(self.calculate_dict_prob() + self.lexicon_alpha), [1])
    return model_score
  
  def calculate_coeff(self, x):
    if self.coeff is None:
      self.coeff = dy.logistic(self.coef_predictor.transform(x))
    return self.coeff
  
  def calculate_dict_prob(self):
    if self.dict_prob is None:
      self.dict_prob = self.lexicon_prob * self.attender.get_last_attention()
    return self.dict_prob
  
  def calc_probs(self, x: dy.Expression) -> dy.Expression:
    model_score = dy.softmax(self.calc_scores(x))
    if self.lexicon_type == 'linear':
      coeff = self.calculate_coeff(x)
      return dy.sum_dim(dy.cmult(coeff, model_score) + dy.cmult((1-coeff), self.calculate_dict_prob()), [1])
    else:
      return model_score

  def calc_log_probs(self, x: dy.Expression) -> dy.Expression:
    if self.lexicon_type == 'linear':
      return dy.log(self.calc_probs(x))
    else:
      return dy.log_softmax(self.calc_scores(x))

  def can_loss_be_derived_from_scores(self):
    return self.lexicon_type == 'bias' and super().is_modifying_softmax_layer()
