import dynet as dy
import numpy as np
from xnmt.settings import settings

from xnmt import batchers, expression_seqs
from xnmt.models import base as models
from xnmt.persistence import serializable_init, Serializable

##### A class for retrieval databases
# This file contains databases used for retrieval.
# At the moment it includes only a standard database that keeps all of the things
# to be retrieved in a list.

class StandardRetrievalDatabase(Serializable):
  """This is a database to be used for retrieval. Its database member"""

  yaml_tag = "!StandardRetrievalDatabase"

  @serializable_init
  def __init__(self, reader, database_file, dev_id_file=None, test_id_file=None):
    self.reader = reader
    self.database_file = database_file
    self.data = list(reader.read_sents(database_file))
    self.indexed = []
    self.dev_id_file = dev_id_file
    self.test_id_file = test_id_file

  def __getitem__(self, indices):
    trg_examples, trg_masks = batchers.pad([self.data[index] for index in indices])
    return batchers.mark_as_batch(trg_examples), trg_masks

##### The actual retriever class
class Retriever(models.ConditionedModel, models.GeneratorModel):
  """
  A template class implementing a retrieval model.
  """

  def calc_loss(self, src, db_idx):
    """Calculate loss based on a database index.

    Args:
      src: The source input.
      db_idx: The correct index in the database to be retrieved.
    Returns:
      An expression representing the loss.
    """
    raise NotImplementedError('calc_loss must be implemented for Retriever subclasses')

  def index_database(self, indices=None):
    """A function that can be called before actually performing retrieval.

    This will perform any necessary pre-processing to make retrieval more efficient.
    If the model is updated, assume that the indexing result is stale and no longer applicable.
    """
    pass

  def generate(self, src):
    """Perform retrieval, trying to get the sentence that most closely matches in the database.

    Args:
      src: The source.
      i: Id of the input
    Returns:
      The ID of the example that most closely matches in the database.
    """
    raise NotImplementedError('generate must be implemented for Retriever subclasses')

  def initialize_generator(self, **kwargs):
    candidates = None
    if kwargs["candidate_id_file"] is not None:
      with open(kwargs["candidate_id_file"], "r") as f:
        candidates = sorted({int(x):1 for x in f}.keys())
    self.index_database(candidates)

class DotProductRetriever(Retriever, Serializable):
  """
  A retriever trains using max-margin methods.
  """

  yaml_tag = '!DotProductRetriever'

  @serializable_init
  def __init__(self, src_embedder, src_encoder, trg_embedder, trg_encoder, database, loss_direction="forward"):
    """Constructor.

    Args:
      src_embedder: A word embedder for the source language
      src_encoder: An encoder for the source language
      trg_embedder: A word embedder for the target language
      trg_encoder: An encoder for the target language
      database: A database of things to retrieve
    """
    self.src_embedder = src_embedder
    self.src_encoder = src_encoder
    self.trg_embedder = trg_embedder
    self.trg_encoder = trg_encoder
    self.database = database
    self.loss_direction = loss_direction

  def exprseq_pooling(self, exprseq):
    # Reduce to vector
    exprseq = expression_seqs.ExpressionSequence(expr_tensor=exprseq.mask.add_to_tensor_expr(exprseq.as_tensor(),-1e10), mask=exprseq.mask)
    if exprseq.expr_tensor is not None:
      if len(exprseq.expr_tensor.dim()[0]) > 1:
        return dy.max_dim(exprseq.expr_tensor, d=1)
      else:
        return exprseq.expr_tensor
    else:
      return dy.emax(exprseq.expr_list)

  def calc_loss(self, src, db_idx, src_mask=None, trg_mask=None):
    src_embeddings = self.src_embedder.embed_sent(src, mask=src_mask)
    self.src_encoder.set_input(src)
    src_encodings = self.exprseq_pooling(self.src_encoder.transduce(src_embeddings))
    trg_batch, trg_mask = self.database[db_idx]
    # print("trg_mask=\n",trg_mask)
    trg_encodings = self.encode_trg_example(trg_batch, mask=trg_mask)
    dim = trg_encodings.dim()
    trg_reshaped = dy.reshape(trg_encodings, (dim[0][0], dim[1]))
    # ### DEBUG
    # trg_npv = trg_reshaped.npvalue()
    # for i in range(dim[1]):
    #   print("--- trg_reshaped {}: {}".format(i,list(trg_npv[:,i])))
    # ### DEBUG
    prod = dy.transpose(src_encodings) * trg_reshaped
    # ### DEBUG
    # prod_npv = prod.npvalue()
    # for i in range(dim[1]):
    #   print("--- prod {}: {}".format(i,list(prod_npv[0].transpose()[i])))
    # ### DEBUG
    id_range = list(range(len(db_idx)))
    # This is ugly:
    if self.loss_direction == "forward":
      prod = dy.transpose(prod)
      loss = dy.sum_batches(dy.hinge_batch(prod, id_range))
    elif self.loss_direction == "bidirectional":
      prod = dy.reshape(prod, (len(db_idx), len(db_idx)))
      loss = dy.sum_elems(
        dy.hinge_dim(prod, id_range, d=0) + dy.hinge_dim(prod, id_range, d=1))
    else:
      raise RuntimeError("Illegal loss direction {}".format(self.loss_direction))

    return loss

  def index_database(self, indices=None):
    # Create the inverted index if necessary
    if indices is None:
      indices = range(len(self.database.data))
      self.database.inverted_index = None
    else:
      self.database.inverted_index = indices
    # Actually index everything
    self.database.indexed = []
    for index in indices:
      item = self.database.data[int(index)]
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      self.database.indexed.append(self.encode_trg_example(item).npvalue())
    # ### DEBUG
    # for i, x in enumerate(self.database.indexed):
    #   print("--- database {}: {}".format(i,list(x)))
    # ### DEBUG
    self.database.indexed = np.stack(self.database.indexed, axis=1)

  def encode_trg_example(self, example, mask=None):
    embeddings = self.trg_embedder.embed_sent(example, mask=mask)
    self.trg_encoder.set_input(example)
    encodings = self.exprseq_pooling(self.trg_encoder.transduce(embeddings))
    return encodings

  def generate(self, src, return_type="idxscore", nbest=10, forced_trg_ids=None):
    if forced_trg_ids is not None: raise NotImplementedError()
    src_embedding = self.src_embedder.embed_sent(src)
    self.src_encoder.set_input(src)
    src_encoding = dy.transpose(self.exprseq_pooling(self.src_encoder.transduce(src_embedding))).npvalue()
    scores = np.dot(src_encoding, self.database.indexed)
    # print("--- scores: {}".format(list(scores[0])))
    kbest = np.argsort(scores, axis=1)[0,-nbest:][::-1]
    # print("--- kbest: {}".format(kbest))
    ids = kbest if self.database.inverted_index is None else [self.database.inverted_index[x] for x in kbest]

    if return_type == "idxscore":
      return [(i,scores[0,x]) for i, x in zip(ids, kbest)]
    elif return_type == "idx":
      return list(ids)
    elif return_type == "score":
      return [scores[0,x] for x in kbest]
    else:
      raise RuntimeError("Illegal return_type to retrieve: {}".format(return_type))

