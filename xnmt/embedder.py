from __future__ import division, generators

from batcher import *
import dynet as dy
from serializer import Serializable
import model_globals
import yaml

class Embedder:
  """
  A parent class that takes in a sent and outputs an embedded sent.
  """

  def embed(self, word):
    """Embed a single word.

    :param word: This will generally be an integer word ID, but could also be something like a string.
    """
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

  def embed_sent(self, sent):
    """Embed a full sentence worth of words.

    :param sent: This will generally be a list of word IDs, but could also be a list of strings or some other format.
    """
    raise NotImplementedError('embed_sent must be implemented in Embedder subclasses')

  @staticmethod
  def from_spec(input_format, vocab_size, emb_dim, model):
    input_format_lower = input_format.lower()
    if input_format_lower == "text":
      return SimpleWordEmbedder(vocab_size, emb_dim, model)
    elif input_format_lower == "contvec":
      return NoopEmbedder(emb_dim, model)
    else:
      raise RuntimeError("Unknown input type {}".format(input_format))

class ExpressionSequence():
  """A class to represent a sequence of expressions.

  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self, **kwargs):
    """Constructor.

    :param expr_list: a python list of expressions
    :param expr_tensor: a tensor where highest dimension are the sequence items
    :raises valueError:
      raises an exception if neither expr_list nor expr_tensor are given,
      or if both have inconsistent length
    """
    self.expr_list = kwargs.pop('expr_list', None)
    self.expr_tensor = kwargs.pop('expr_tensor', None)
    if not (self.expr_list or self.expr_tensor):
      raise ValueError("must provide expr_list or expr_tensor")
    if self.expr_list and self.expr_tensor:
      if len(self.expr_list) != self.expr_tensor.dim()[0][0]:
        raise ValueError("expr_list and expr_tensor must be of same length")

  def __len__(self):
    """Return length.

    :returns: length of sequence
    """
    return len(self.expr_list) if self.expr_list else self.expr_tensor.dim()[0][0]

  def __iter__(self):
    """Return iterator.

    :returns: iterator over the sequence; results in explicit conversion to list
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return iter(self.expr_list)

  def __getitem__(self, key):
    """Get a single item.

    :returns: sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else: return dy.pick(self.expr_tensor, key)

  def as_tensor(self):
    """Get a tensor.
    :returns: the whole sequence as a tensor expression.
    """
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate(list(map(lambda x:dy.transpose(x), self)))
    return self.expr_tensor

class SimpleWordEmbedder(Embedder, Serializable):
  """
  Simple word embeddings via lookup.
  """

  yaml_tag = u'!SimpleWordEmbedder'

  def __init__(self, vocab_size, emb_dim = None):
    self.vocab_size = vocab_size
    if emb_dim is None: emb_dim = model_globals.get("default_layer_dim")
    self.emb_dim = emb_dim
    self.embeddings = model_globals.get("model").add_lookup_parameters((vocab_size, emb_dim))

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return self.embeddings[x]
    # minibatch mode
    else:
      return self.embeddings.batch(x)

  def embed_sent(self, sent):
    # single mode
    if not Batcher.is_batch_sent(sent):
      embeddings = [self.embed(word) for word in sent]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sent[0])):
        embeddings.append(self.embed(Batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])))

    return ExpressionSequence(expr_list=embeddings)

class NoopEmbedder(Embedder, Serializable):
  """
  This embedder performs no lookups but only passes through the inputs.

  Normally, the input is an Input object, which is converted to an expression.

  We can also input an ExpressionSequence, which is simply returned as-is.
  This is useful e.g. to stack several encoders, where the second encoder performs no
  lookups.
  """

  yaml_tag = u'!NoopEmbedder'
  def __init__(self, emb_dim):
    self.emb_dim = emb_dim

  def embed(self, x):
    if isinstance(x, dy.Expression): return x
    # single mode
    if not Batcher.is_batch_word(x):
      return dy.inputVector(x)
    # minibatch mode
    else:
      return dy.inputTensor(x, batched=True)

  def embed_sent(self, sent):
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    if isinstance(sent, ExpressionSequence):
      return sent

    batched = Batcher.is_batch_sent(sent)
    first_sent = sent[0] if batched else sent
    if hasattr(first_sent, "get_array"):
      if not batched:
        return ExpressionSequence(expr_tensor=dy.inputTensor(sent.get_array(), batched=False))
      else:
        return ExpressionSequence(expr_tensor=dy.inputTensor(map(lambda s: s.get_array(), sent), batched=True))
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sent]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(Batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])))
      return ExpressionSequence(expr_list=embeddings)

