from __future__ import division, generators

from batcher import *
import dynet as dy
from serializer import Serializable
from expression_sequence import ExpressionSequence
import model_globals
import yaml

class Embedder:
  """
  An embedder takes in word IDs and outputs continuous vectors.
  
  This can be done on a word-by-word basis, or over a sequence.
  """

  def embed(self, word):
    """Embed a single word.

    :param word: This will generally be an integer word ID, but could also be something like a string. It could
      also be batched, in which case the input will be a list of integers or other things.
    :returns: A DyNet Expression corresponding to the embedding of the word(s).
    """
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

  def embed_sent(self, sent):
    """Embed a full sentence worth of words.

    :param sent: This will generally be a list of word IDs, but could also be a list of strings or some other format.
      It could also be batched, in which case it will be a list of list.
    :returns: An ExpressionSequence representing vectors of each word in the input.
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

