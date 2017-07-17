from __future__ import division, generators

import six
import dynet as dy
from batcher import *
from search_strategy import *
from vocab import Vocab
from serializer import Serializable, DependentInitParam
from train_test_interface import TrainTestInterface
import numpy as np
import os

##### A class for retrieval databases

# This file contains databases used for retrieval.
# At the moment it includes only a standard database that keeps all of the things
# to be retrieved in a list.

class StandardRetrievalDatabase(Serializable):
  """This is a database to be used for retrieval. Its database member"""

  yaml_tag = u"!StandardRetrievalDatabase"

  def __init__(self, reader, database_file):
    self.reader = reader
    self.database_file = database_file
    self.data = list(reader.read_sents(database_file))
    self.indexed = []

  def __getitem__(self, indices):
    return Batcher.mark_as_batch(Batcher.pad([self.data[index] for index in indices]))

##### The actual retriever class

class Retriever(TrainTestInterface):
  '''
  A template class implementing a retrieval model.
  '''

  def calc_loss(self, src, db_idx):
    '''Calculate loss based on a database index.

    :param src: The source input.
    :param db_idx: The correct index in the database to be retrieved.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Retriever subclasses')

  def index_database(self):
    '''A function that can be called before actually performing retrieval.

    This will perform any necessary pre-processing to make retrieval more efficient.
    If the model is updated, assume that the indexing result is stale and no longer applicable.
    '''
    pass

  def retrieve(self, src):
    '''Perform retrieval, trying to get the sentence that most closely matches in the database.

    :param src: The source.
    :returns: The ID of the example that most closely matches in the database.
    '''
    raise NotImplementedError('retrieve must be implemented for Retriever subclasses')

  def set_train(self, val):
    for component in self.get_train_test_components():
      Retriever.set_train_recursive(component, val)

  @staticmethod
  def set_train_recursive(component, val):
    component.set_train(val)
    for sub_component in component.get_train_test_components():
      Retriever.set_train_recursive(sub_component, val)


class DotProductRetriever(Retriever, Serializable):
  '''
  A retriever trains using max-margin methods.
  '''

  yaml_tag = u'!DotProductRetriever'


  def __init__(self, src_embedder, src_encoder, trg_embedder, trg_encoder, database):
    '''Constructor.

    :param src_embedder: A word embedder for the source language
    :param src_encoder: An encoder for the source language
    :param trg_embedder: A word embedder for the target language
    :param trg_encoder: An encoder for the target language
    :param database: A database of things to retrieve
    '''
    self.src_embedder = src_embedder
    self.src_encoder = src_encoder
    self.trg_embedder = trg_embedder
    self.trg_encoder = trg_encoder
    self.database = database

  def get_train_test_components(self):
    return [self.src_encoder, self.trg_encoder]

  def exprseq_pooling(self, exprseq):
    # Reduce to vector
    if exprseq.expr_tensor != None:
      if len(exprseq.expr_tensor.dim()[0]) > 1:
        return dy.max_dim(exprseq.expr_tensor, d=1)
      else:
        return exprseq.expr_tensor
    else:
      return dy.emax(exprseq.expr_list)

  def calc_loss(self, src, db_idx):
    src_embeddings = self.src_embedder.embed_sent(src)
    src_encodings = self.exprseq_pooling(self.src_encoder.transduce(src_embeddings))
    trg_encodings = self.encode_trg_example(self.database[db_idx])

    prod = dy.transpose(dy.transpose(src_encodings) * trg_encodings)
    loss = dy.sum_batches(dy.hinge_batch(prod, list(six.moves.range(len(db_idx)))))
    print(loss.npvalue())
    return loss

  def index_database(self, subsample_file='examples/data/flickr_subsample_index.txt'):
    self.database.indexed = []
    if subsample_file != None:
      indices = list(np.loadtxt(subsample_file))
    else:
      indices = range(len(self.database.data))
    for index in indices:
      item = self.database.data[int(index)]
      dy.renew_cg()
      self.database.indexed.append(self.encode_trg_example(item).npvalue())
    self.database.indexed = np.concatenate(self.database.indexed, axis=1)

  def encode_trg_example(self, example):
    embeddings = self.trg_embedder.embed_sent(example)
    encodings = self.exprseq_pooling(self.trg_encoder.transduce(embeddings))
    dim = encodings.dim()
    return dy.reshape(encodings, (dim[0][0], dim[1]))

  def retrieve(self, src, return_type="idxscore", nbest=5):
    src_embedding = self.src_embedder.embed_sent(src)
    src_encoding = dy.transpose(self.exprseq_pooling(self.src_encoder.transduce(src_embedding))).npvalue()

    scores = np.dot(src_encoding, self.database.indexed)
    kbest = np.argsort(scores, axis=1)[0,-nbest:][::-1]
    if return_type == "idxscore":
      return [(x,scores[0,x]) for x in kbest]
    elif return_type == "idx":
      return list(kbest)
    elif return_type == "score":
      return [scores[0,x] for x in kbest]
    else:
      raise RuntimeError("Illegal return_type to retrieve: {}".format(return_type))
