from __future__ import division, generators

from batcher import *

class Embedder:
  '''
  A template class to embed a word or token.
  '''

  '''
  Takes a string or word ID and returns its embedding.
  '''

  def embed(self, x):
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

class EmbeddedSentence():
  '''
  Represents an embedded sentence.
  '''
  def __len__(self): raise NotImplementedError("__len__() must be implemented by EmbeddedSentence subclasses")
  def __iter__(self): raise NotImplementedError("__iter__() must be implemented by EmbeddedSentence subclasses")
  def __getitem__(self, key): raise NotImplementedError("__getitem__() must be implemented by EmbeddedSentence subclasses")
      
class ListEmbeddedSentence(list, EmbeddedSentence):
  '''
  Represents an embedded sentence as a list of (e.g. vector-) expressions
  '''
#  def __init__(self, l):
#    super(list, self).__init__(l)

class TensorExprEmbeddedSentence(EmbeddedSentence):
  '''
  Represents an embedded sentence as a single tensor expression, where words correspond to the first dimension.
  '''
  def __init__(self, tensorExpr): self.tensorExpr = tensorExpr
  def __len__(self): return self.tensorExpr.dim()[0][0]
  def __iter__(self): return iter([self[i] for i in range(len(self))])
  def __getitem__(self, key): return dy.pick(self.tensorExpr, key) 
  def get_tensor_repr(self): return self.tensorExpr
      
class SimpleWordEmbedder(Embedder):
  'Simple word embeddings'

  def __init__(self, vocab_size, emb_dim, model):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.embeddings = model.add_lookup_parameters((vocab_size, emb_dim))
    self.serialize_params = [vocab_size, emb_dim, model]

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return self.embeddings[x]
    # minibatch mode
    else:
      return self.embeddings.batch(x)

  def embed_sentence(self, sentence):
    # single mode
    if not Batcher.is_batch_sentence(sentence):
      embeddings = [self.embed(word) for word in sentence]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sentence[0])):
        embeddings.append(self.embed(Batcher.mark_as_batch([single_sentence[word_i] for single_sentence in sentence])))

    return ListEmbeddedSentence(embeddings)

class FeatVecNoopEmbedder(Embedder):
  def __init__(self, emb_dim, model):
    self.emb_dim = emb_dim
    self.serialize_params = [emb_dim, model]

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return dy.inputVector(x)
    # minibatch mode
    else:
      return dy.inputTensor(x, batched=True)

  def embed_sentence(self, sentence):
    batched = Batcher.is_batch_sentence(sentence)
    first_sent = sentence[0] if batched else sentence
    if hasattr(first_sent, "get_array"):
      return TensorExprEmbeddedSentence(dy.inputTensor(map(lambda s: s.get_array(), sentence), batched=batched))
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sentence]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(Batcher.mark_as_batch([single_sentence[word_i] for single_sentence in sentence])))
      return ListEmbeddedSentence(embeddings)

