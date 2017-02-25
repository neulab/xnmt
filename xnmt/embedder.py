class Embedder:
  '''
  A template class to embed a word or token.
  '''

  '''
  Takes a string or word ID and returns its embedding.
  '''

  def embed(self, x):
    raise NotImplementedError('embed must be implemented in Embedder subclasses')


class SimpleWordEmbedder(Embedder):
  'Simple word embeddings'

  def __init__(self, vocab_size, emb_dim, model):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.embeddings = model.add_lookup_parameters((vocab_size, emb_dim))

  def embed(self, x):
    return self.embeddings[x]

  def embed_batch(self, x):
    return self.embeddings.batch(x)