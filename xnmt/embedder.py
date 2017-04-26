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

  @staticmethod
  def from_spec(input_format, vocab_size, emb_dim, model):
    input_format_lower = input_format.lower()
    if input_format_lower == "text":
      return SimpleWordEmbedder(vocab_size, emb_dim, model)
    elif input_format_lower == "contvec":
      return NoopEmbedder(emb_dim, model)
    else:
      raise RuntimeError("Unknown input type {}".format(input_format))

class EmbeddedSentence():
  """
  A template class to represent an embedded sentence.
  """
  def __len__(self): raise NotImplementedError("__len__() must be implemented by EmbeddedSentence subclasses")
  def __iter__(self): raise NotImplementedError("__iter__() must be implemented by EmbeddedSentence subclasses")
  def __getitem__(self, key): raise NotImplementedError("__getitem__() must be implemented by EmbeddedSentence subclasses")
      
class ListEmbeddedSentence(list, EmbeddedSentence):
  """
  Represents an embedded sentence as a list of expressions.
  """
  pass # only inherit from list

class TensorExprEmbeddedSentence(EmbeddedSentence):
  """
  Represents an embedded sentence as a single tensor expression, where words correspond to the first dimension.
  """
  def __init__(self, tensorExpr): self.tensorExpr = tensorExpr
  def __len__(self): return self.tensorExpr.dim()[0][0]
  def __iter__(self): return iter([self[i] for i in range(len(self))])
  def __getitem__(self, key): return dy.pick(self.tensorExpr, key) 
  def get_tensor_repr(self): return self.tensorExpr
      
class SimpleWordEmbedder(Embedder):
  """
  Simple word embeddings via lookup.
  """

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

class NoopEmbedder(Embedder):
  """
  This embedder performs no lookups but only passes through the inputs.
  
  Normally, then input is an Input object, which is converted to an expression.
  
  We can also input an expression or EmbeddedSentence , which is simply returned as-is.
  This is useful e.g. to stack several encoders, where the second encoder performs no
  lookups.
  """
  def __init__(self, emb_dim, model):
    self.emb_dim = emb_dim
    self.serialize_params = [emb_dim, model]

  def embed(self, x):
    if isinstance(x, dy.Expression): return x
    # single mode
    if not Batcher.is_batch_word(x):
      return dy.inputVector(x)
    # minibatch mode
    else:
      return dy.inputTensor(x, batched=True)

  def embed_sentence(self, sentence):
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    if isinstance(sentence, EmbeddedSentence):
      return sentence
    elif isinstance(sentence, dy.Expression):
      return TensorExprEmbeddedSentence(sentence)
    elif type(sentence)==list and type(sentence[0])==dy.Expression:
      return ListEmbeddedSentence(sentence)
    
    batched = Batcher.is_batch_sentence(sentence)
    first_sent = sentence[0] if batched else sentence
    if hasattr(first_sent, "get_array"):
      if not batched:
        return TensorExprEmbeddedSentence(dy.inputTensor(sentence.get_array(), batched=False))
      else:
        return TensorExprEmbeddedSentence(dy.inputTensor(map(lambda s: s.get_array(), sentence), batched=True))
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sentence]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(Batcher.mark_as_batch([single_sentence[word_i] for single_sentence in sentence])))
      return ListEmbeddedSentence(embeddings)

