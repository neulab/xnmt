from __future__ import division, generators

import dynet as dy
import batcher
import model_globals
import six

from serializer import Serializable
from expression_sequence import ExpressionSequence, LazyNumpyExpressionSequence

class Embedder(object):
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

  def embed_sent(self, sent, mask=None):
    """Embed a full sentence worth of words.

    :param sent: This will generally be a list of word IDs, but could also be a list of strings or some other format.
      It could also be batched, in which case it will be a list of list.
    :returns: An ExpressionSequence representing vectors of each word in the input.
    """
    raise NotImplementedError('embed_sent must be implemented in Embedder subclasses')

class SimpleWordEmbedder(Embedder, Serializable):
  """
  Simple word embeddings via lookup.
  """

  yaml_tag = u'!SimpleWordEmbedder'

  def __init__(self, vocab_size, emb_dim = None, weight_noise = None):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim or model_globals.get("default_layer_dim")
    self.weight_noise = weight_noise or model_globals.get("weight_noise")
    self.embeddings = model_globals.dynet_param_collection.param_col.add_lookup_parameters((vocab_size, emb_dim))

  def embed(self, x):
    # single mode
    if not batcher.is_batched(x):
      ret = self.embeddings[x]
    # minibatch mode
    else:
      ret = self.embeddings.batch(x)
    if self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

  def embed_sent(self, sent, mask=None):
    # single mode
    if not batcher.is_batched(sent):
      embeddings = [self.embed(word) for word in sent]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sent[0])):
        embeddings.append(self.embed(batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])))

    return ExpressionSequence(expr_list=embeddings, mask=mask)

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
    return dy.inputTensor(x, batched=batcher.is_batched(x))

  def embed_sent(self, sent, mask=None):
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    if isinstance(sent, ExpressionSequence):
      return sent
    batched = batcher.is_batched(sent)
    first_sent = sent[0] if batched else sent
    if hasattr(first_sent, "get_array"):
      if not batched:
        return LazyNumpyExpressionSequence(lazy_data=sent.get_array())
      else:
        return LazyNumpyExpressionSequence(lazy_data=batcher.mark_as_batch(
                                            six.moves.map(lambda s: s.get_array(), sent)),
                                           mask=mask)
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sent]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])))
      return ExpressionSequence(expr_list=embeddings, mask=mask)

