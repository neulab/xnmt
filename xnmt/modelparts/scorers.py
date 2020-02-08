from typing import List, Tuple, Union, Optional
import numbers

import numpy as np

import xnmt
from xnmt import batchers, input_readers, param_collections, param_initializers, vocabs, logger, tensor_tools as tt
from xnmt.modelparts import transforms
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.events import handle_xnmt_event, register_xnmt_handler

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch
  import torch.nn.functional as F
  import torch.nn

def find_best_k(scores: np.ndarray, k: numbers.Integral) -> Tuple[np.ndarray, np.ndarray]:
  """
  Args:
    scores: numpy array of dim (#classes, batch_size)
    k: integer
  Returns:
    tuple: indices of top words [k, batch_size], top scores [k, batch_size]
  """
  k = min(len(scores), k)
  top_words = np.argpartition(scores, -k, axis=0)[-k:]

  if len(scores.shape) > 1:
    assert top_words.shape == (k, scores.shape[1]), \
      'top_words has shape %s, expected (%d, %d)' % (str(top_words.shape), k, scores.shape[1])
    # top_words is (k, batch_size)
    # scores is (#classes, batch_size)
    top_scores = []
    for i in range(top_words.shape[1]):
      top_scores.append(scores[top_words[:, i], i])
    top_scores = np.array(top_scores).T
  else:
    assert top_words.shape == (k,)
    top_scores = scores[top_words]
  return top_words, top_scores

class Scorer(object):
  """
  A template class of things that take in a vector and produce a
  score over discrete output items.
  """

  def calc_scores(self, x: tt.Tensor) -> tt.Tensor:
    """
    Calculate the score of each discrete decision, where the higher
    the score is the better the model thinks a decision is. These
    often correspond to unnormalized log probabilities.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_scores must be implemented by subclasses of Scorer')

  def best_k(self, x: tt.Tensor, k: numbers.Integral, normalize_scores: bool = False):
    """
    Returns a list of the k items with the highest scores. The items may not be
    in sorted order.

    Args:
      x: The vector used to make the prediction
      k: Number of items to return
      normalize_scores: whether to normalize the scores
    """
    raise NotImplementedError('best_k must be implemented by subclasses of Scorer')

  def sample(self, x: tt.Tensor, n: numbers.Integral):
    """
    Return samples from the scores that are treated as probability distributions.
    """
    raise NotImplementedError('sample must be implemented by subclasses of Scorer')

  def calc_probs(self, x: tt.Tensor) -> tt.Tensor:
    """
    Calculate the normalized probability of a decision.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_prob must be implemented by subclasses of Scorer')

  def calc_log_probs(self, x: tt.Tensor) -> tt.Tensor:
    """
    Calculate the log probability of a decision
    
    log(calc_prob()) == calc_log_prob()

    Both functions exist because it might help save memory.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_log_prob must be implemented by subclasses of Scorer')

  def calc_loss(self, x: tt.Tensor, y: Union[int, List[int]]) -> tt.Tensor:
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

@xnmt.require_dynet
class SoftmaxDynet(Scorer, Serializable):
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
    self.input_dim = input_dim
    self.output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.label_smoothing = label_smoothing
    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or transforms.Linear(
                                                              input_dim=self.input_dim, output_dim=self.output_dim,
                                                              param_init=param_init, bias_init=bias_init))

  def calc_scores(self, x: tt.Tensor) -> tt.Tensor:
    return self.output_projector.transform(x)

  def best_k(self, x: tt.Tensor, k: numbers.Integral, normalize_scores: bool = False):
    scores_expr = self.calc_log_probs(x) if normalize_scores else self.calc_scores(x)
    scores = scores_expr.npvalue()
    return find_best_k(scores, k)

  def sample(self, x: tt.Tensor, n: numbers.Integral, temperature: numbers.Real=1.0):
    assert temperature != 0.0
    scores_expr = self.calc_log_probs(x)
    if temperature != 1.0:
      scores_expr *= 1.0 / temperature
      scores = dy.softmax(scores_expr).npvalue()
    else:
      scores = dy.exp(scores_expr).npvalue()

    # Numpy is very picky. If the sum is off even by 1e-8 it complains.
    scores /= sum(scores)

    a = range(scores.shape[0])
    samples = np.random.choice(a, (n,), replace=True, p=scores)

    r = []
    for word in samples:
      r.append((word, dy.pick(scores_expr, word)))
    return r

  def _can_loss_be_derived_from_scores(self):
    """
    This method can be used to determine whether dy.pickneglogsoftmax can be used to quickly calculate the loss value.
    If False, then the calc_loss method should (1) calc log_softmax, (2) perform necessary modification, (3) pick the loss
    """
    return self.label_smoothing == 0.0

  def calc_loss(self, x: tt.Tensor, y: Union[numbers.Integral, List[numbers.Integral]]) -> tt.Tensor:
    if self._can_loss_be_derived_from_scores():
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

  def calc_probs(self, x: tt.Tensor) -> tt.Tensor:
    return dy.softmax(self.calc_scores(x))

  def calc_log_probs(self, x: tt.Tensor) -> tt.Tensor:
    return dy.log_softmax(self.calc_scores(x))

@xnmt.require_dynet
class LexiconSoftmax(SoftmaxDynet, Serializable):
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

  def calc_scores(self, x: tt.Tensor) -> tt.Tensor:
    model_score = self.output_projector.transform(x)
    if self.lexicon_type == 'bias':
      model_score += dy.sum_dim(dy.log(self.calculate_dict_prob(x) + self.lexicon_alpha), [1])
    return model_score

  def calculate_coeff(self, x):
    if self.coeff is None:
      self.coeff = dy.logistic(self.coef_predictor.transform(x))
    return self.coeff

  def calculate_dict_prob(self, x):
    if self.dict_prob is None:
      self.dict_prob = self.lexicon_prob * self.attender.calc_attention(x)
    return self.dict_prob

  def calc_probs(self, x: tt.Tensor) -> tt.Tensor:
    model_score = dy.softmax(self.calc_scores(x))
    if self.lexicon_type == 'linear':
      coeff = self.calculate_coeff(x)
      return dy.sum_dim(dy.cmult(coeff, model_score) + dy.cmult((1-coeff), self.calculate_dict_prob(x)), [1])
    else:
      return model_score

  def calc_log_probs(self, x: tt.Tensor) -> tt.Tensor:
    if self.lexicon_type == 'linear':
      return dy.log(self.calc_probs(x))
    else:
      return dy.log_softmax(self.calc_scores(x))

  def _can_loss_be_derived_from_scores(self):
    return self.lexicon_type == 'bias' and super()._can_loss_be_derived_from_scores()

@xnmt.require_torch
class SoftmaxTorch(Scorer, Serializable):
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
    self.input_dim = input_dim
    self.output_dim = self._choose_vocab_size(vocab_size, vocab, trg_reader)
    self.label_smoothing = label_smoothing
    self.output_projector = self.add_serializable_component("output_projector", output_projector,
                                                            lambda: output_projector or transforms.Linear(
                                                              input_dim=self.input_dim, output_dim=self.output_dim,
                                                              param_init=param_init, bias_init=bias_init))

  def calc_scores(self, x: tt.Tensor) -> tt.Tensor:
    return self.output_projector.transform(x)

  def best_k(self, x: tt.Tensor, k: numbers.Integral, normalize_scores: bool = False):
    scores_expr = self.calc_log_probs(x) if normalize_scores else self.calc_scores(x)
    scores = scores_expr.cpu().transpose(0,1).data.numpy()
    if scores.shape[1]==1: scores = np.resize(scores, (scores.shape[0],))
    return find_best_k(scores, k)

  def sample(self, x: tt.Tensor, n: numbers.Integral, temperature: numbers.Real=1.0):
    raise NotImplementedError()

  def calc_loss(self, x: tt.Tensor, y: Union[numbers.Integral, List[numbers.Integral]]) -> tt.Tensor:
    if self.label_smoothing:
      # following this implementation:
      # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
      pred = self.calc_scores(x)
      eps = self.label_smoothing
      n_class = self.output_dim
      gold = torch.tensor(y).to(xnmt.device)
      one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1,1), 1)
      # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class  # original version does not add up to 1
      one_hot = one_hot * (1 - eps) + eps / n_class
      log_prb = F.log_softmax(pred, dim=1)
      return -(torch.matmul(one_hot.unsqueeze(1), log_prb.unsqueeze(2)).squeeze(2)) # neg dot product
    else:
      # scores = torch.nn.LogSoftmax(dim=-1)(self.calc_scores(x))
      # return F.nll_loss(input=scores, target=torch.tensor(y).to(xnmt.device), reduction='none')
      if np.isscalar(y): y = [y]
      return F.cross_entropy(self.calc_scores(x), target=torch.tensor(y,dtype=torch.long).to(xnmt.device), reduction='none')

  def calc_probs(self, x: tt.Tensor) -> tt.Tensor:
    return torch.nn.Softmax(dim=-1)(self.calc_scores(x))

  def calc_log_probs(self, x: tt.Tensor) -> tt.Tensor:
    return torch.nn.LogSoftmax(dim=-1)(self.calc_scores(x))

Softmax = xnmt.resolve_backend(SoftmaxDynet, SoftmaxTorch)