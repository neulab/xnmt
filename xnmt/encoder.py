import dynet as dy
from batcher import *


class Encoder:
  '''
  A template class to encode an input.
  '''

  '''
  Takes an Input and returns an EncodedInput.
  '''

  def encode(self, x):
    raise NotImplementedError('encode must be implemented in Encoder subclasses')


class BiLstmEncoder(Encoder):
  def __init__(self, layers, output_dim, embedder, model):
    self.embedder = embedder
    input_dim = embedder.emb_dim
    self.encoder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.LSTMBuilder)
    self.model_lookup = model.add_lookup_parameters((embedder.vocab_size, embedder.emb_dim))

  def encode(self, sentence):
    # single mode
    if not Batcher.is_batch_sentence(sentence):
      embeddings = [self.embedder.embed(word) for word in sentence]

    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sentence[0])):
        embeddings.append(self.embedder.embed_batch([single_sentence[word_i] for single_sentence in sentence]))

    return self.encoder.transduce(embeddings)
