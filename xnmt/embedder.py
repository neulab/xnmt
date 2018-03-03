import logging
logger = logging.getLogger('xnmt')
import io

import numpy as np
import dynet as dy

import xnmt.batcher
from xnmt.initializer import LeCunUniform as linear_init
from xnmt.events import register_handler, handle_xnmt_event
from xnmt.serialize.serializable import Serializable
from xnmt.serialize.tree_tools import Ref, Path
from xnmt.expression_sequence import ExpressionSequence, LazyNumpyExpressionSequence
from xnmt.linear import Linear

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

  def embed_sent(self, sent):
    """Embed a full sentence worth of words. By default, just do a for loop.

    :param sent: This will generally be a list of word IDs, but could also be a list of strings or some other format.
      It could also be batched, in which case it will be a (possibly masked) Batch object
    :returns: An ExpressionSequence representing vectors of each word in the input.
    """
    # single mode
    if not xnmt.batcher.is_batched(sent):
      embeddings = [self.embed(word) for word in sent]
    # minibatch mode
    else:
      embeddings = []
      seq_len = len(sent[0])
      for single_sent in sent: assert len(single_sent)==seq_len
      for word_i in range(seq_len):
        batch = xnmt.batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])
        embeddings.append(self.embed(batch))

    return ExpressionSequence(expr_list=embeddings, mask=sent.mask if xnmt.batcher.is_batched(sent) else None)

  def choose_vocab(self, vocab, yaml_path, src_reader, trg_reader):
    """Choose the vocab for the embedder basd on the passed arguments

    This is done in order of priority of vocab, model+yaml_path
    """
    if vocab != None:
      return len(vocab)
    elif "src_embedder" in yaml_path:
      if src_reader == None or src_reader.vocab == None:
        raise ValueError("Could not determine src_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "trg_embedder" in yaml_path or "vocab_projector" in yaml_path:
      if trg_reader == None or trg_reader.vocab == None:
        raise ValueError("Could not determine trg_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError("Attempted to determine vocab size of {} (path: {}), but path was not src_embedder, trg_embedder, or vocab_projector, so it could not determine what part of the model to use. Please set vocab_size or vocab explicitly.".format(self.__class__, yaml_path))

  def choose_vocab_size(self, vocab_size, vocab, yaml_path, src_reader, trg_reader):
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path
    """
    if vocab_size != None:
      return vocab_size
    elif vocab != None:
      return len(vocab)
    elif "src_embedder" in yaml_path:
      if src_reader == None or src_reader.vocab == None:
        raise ValueError("Could not determine src_embedder's size. Please set its vocab_size or vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "trg_embedder" in yaml_path or "vocab_projector" in yaml_path:
      if trg_reader == None or trg_reader.vocab == None:
        raise ValueError("Could not determine target embedder's size. Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError("Attempted to determine vocab size of {} (path: {}), but path was not src_embedder, trg_embedder, or vocab_projector, so it could not determine what part of the model to use. Please set vocab_size or vocab explicitly.".format(self.__class__, yaml_path))

class DenseWordEmbedder(Embedder, Linear, Serializable):
  """
  Word embeddings via full matrix
  """
  yaml_tag = "!DenseWordEmbedder"
  def __init__(self, exp_global=Ref(Path("exp_global")), emb_dim=None, weight_noise=None, word_dropout=0.0,
               fix_norm=None, vocab_size=None, vocab=None, yaml_path=None,
               src_reader=Ref(path=Path("model.src_reader"), required=False),
               trg_reader=Ref(path=Path("model.trg_reader"), required=False)):
    register_handler(self)
    self.fix_norm = fix_norm
    self.weight_noise = weight_noise or exp_global.weight_noise
    self.word_dropout = word_dropout
    self.emb_dim = emb_dim or exp_global.default_layer_dim
    self.dynet_param_collection = exp_global.dynet_param_collection
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.embeddings = self.dynet_param_collection.param_col.add_parameters((self.vocab_size, self.emb_dim))
    self.bias = self.dynet_param_collection.param_col.add_parameters((self.vocab_size))

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.word_id_mask = None

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def embed(self, x):
    if self.train and self.word_dropout > 0.0 and self.word_id_mask is None:
      batch_size = len(x) if xnmt.batcher.is_batched(x) else 1
      self.word_id_mask = [set(np.random.choice(self.vocab_size, int(self.vocab_size * self.word_dropout), replace=False)) for _ in range(batch_size)]
    emb_e = dy.parameter(self.embeddings)
    # single mode
    if not xnmt.batcher.is_batched(x):
      if self.train and self.word_id_mask and x in self.word_id_mask[0]:
        ret = dy.zeros((self.emb_dim,))
      else:
        ret = dy.pick(emb_e, index=x)
        if self.fix_norm != None:
          ret = dy.cdiv(ret, dy.l2_norm(ret))
          if self.fix_norm != 1:
            ret *= self.fix_norm
    # minibatch mode
    else:
      ret = dy.concatenate_to_batch([dy.pick(emb_e, index=xi) for xi in x])
      if self.fix_norm != None:
        ret = dy.cdiv(ret, dy.l2_norm(ret))
        if self.fix_norm != 1:
          ret *= self.fix_norm
      if self.train and self.word_id_mask and any(x[i] in self.word_id_mask[i] for i in range(len(x))):
        dropout_mask = dy.inputTensor(np.transpose([[0.0]*self.emb_dim if x[i] in self.word_id_mask[i] else [1.0]*self.emb_dim for i in range(len(x))]), batched=True)
        ret = dy.cmult(ret, dropout_mask)
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

  def __call__(self, input_expr):
    W1 = dy.parameter(self.embeddings)
    b1 = dy.parameter(self.bias)
    return dy.affine_transform([b1, W1, input_expr])


class SimpleWordEmbedder(Embedder, Serializable):
  """
  Simple word embeddings via lookup.
  """

  yaml_tag = '!SimpleWordEmbedder'

  def __init__(self, exp_global=Ref(Path("exp_global")), emb_dim=None, weight_noise=None, word_dropout=0.0,
               fix_norm=None, init=None, vocab_size = None, vocab = None, yaml_path = None,
               src_reader = Ref(path=Path("model.src_reader"), required=False), trg_reader = Ref(path=Path("model.trg_reader"), required=False)):
    """
    :param emb_dim:
    :param weight_noise: apply Gaussian noise with given standard deviation to embeddings
    :param word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    :param fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
    """
    register_handler(self)
    self.emb_dim = emb_dim or exp_global.default_layer_dim
    self.weight_noise = weight_noise or exp_global.weight_noise
    self.word_dropout = word_dropout
    self.fix_norm = fix_norm
    self.word_id_mask = None
    self.train = False
    self.dynet_param_collection = exp_global.dynet_param_collection
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    if init == 'LeCunUniform':
      init = linear_init(self.vocab_size)
    self.embeddings = self.dynet_param_collection.param_col.add_lookup_parameters((self.vocab_size, self.emb_dim),
                                                                                  init=init)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.word_id_mask = None

  def embed(self, x):
    if self.train and self.word_dropout > 0.0 and self.word_id_mask is None:
      batch_size = len(x) if xnmt.batcher.is_batched(x) else 1
      self.word_id_mask = [set(np.random.choice(self.vocab_size, int(self.vocab_size * self.word_dropout), replace=False)) for _ in range(batch_size)]
    # single mode
    if not xnmt.batcher.is_batched(x):
      if self.train and self.word_id_mask and x in self.word_id_mask[0]:
        ret = dy.zeros((self.emb_dim,))
      else:
        ret = self.embeddings[x]
        if self.fix_norm != None:
          ret = dy.cdiv(ret, dy.l2_norm(ret))
          if self.fix_norm != 1:
            ret *= self.fix_norm
    # minibatch mode
    else:
      ret = self.embeddings.batch(x)
      if self.fix_norm != None:
        ret = dy.cdiv(ret, dy.l2_norm(ret))
        if self.fix_norm != 1:
          ret *= self.fix_norm
      if self.train and self.word_id_mask and any(x[i] in self.word_id_mask[i] for i in range(len(x))):
        dropout_mask = dy.inputTensor(np.transpose([[0.0]*self.emb_dim if x[i] in self.word_id_mask[i] else [1.0]*self.emb_dim for i in range(len(x))]), batched=True)
        ret = dy.cmult(ret, dropout_mask)
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret
  
  def __call__(self, input_expr):
    return dy.transpose(dy.parameter(self.embeddings)) * input_expr


    
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
    return dy.inputTensor(x, batched=xnmt.batcher.is_batched(x))

  def embed_sent(self, sent):
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    if isinstance(sent, ExpressionSequence):
      return sent
    batched = xnmt.batcher.is_batched(sent)
    first_sent = sent[0] if batched else sent
    if hasattr(first_sent, "get_array"):
      if not batched:
        return LazyNumpyExpressionSequence(lazy_data=sent.get_array())
      else:
        return LazyNumpyExpressionSequence(lazy_data=xnmt.batcher.mark_as_batch(
                                           map(lambda s: s.get_array(), sent)),
                                           mask=sent.mask)
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sent]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(xnmt.batcher.mark_as_batch([single_sent[word_i] for single_sent in sent])))
      return ExpressionSequence(expr_list=embeddings, mask=sent.mask)


class PretrainedSimpleWordEmbedder(SimpleWordEmbedder):
  """
  Simple word embeddings via lookup. Initial pretrained embeddings must be supplied in FastText text format.
  """

  yaml_tag = '!PretrainedSimpleWordEmbedder'

  def __init__(self, filename, emb_dim=None, weight_noise=None, word_dropout=0.0, fix_norm = None, vocab = None, yaml_path = None,
               src_reader = Ref(path=Path("model.src_reader"), required=False), trg_reader = Ref(path=Path("model.trg_reader"), required=False), exp_global=Ref(Path("exp_global"))):
    """
    :param filename: Filename for the pretrained embeddings
    :param weight_noise: apply Gaussian noise with given standard deviation to embeddings
    :param word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    :param vocab: a `Vocab` object containing the vocabulary for the experiment
    """
    self.emb_dim = emb_dim or exp_global.default_layer_dim
    self.weight_noise = weight_noise or exp_global.weight_noise
    self.word_dropout = word_dropout
    self.word_id_mask = None
    self.train = False
    self.fix_norm = fix_norm
    self.pretrained_filename = filename
    self.dynet_param_collection = exp_global.dynet_param_collection
    self.vocab = self.choose_vocab(vocab, yaml_path, src_reader, trg_reader)
    self.vocab_size = len(vocab)
    with io.open(self.pretrained_filename, encoding='utf-8') as embeddings_file:
      total_embs, in_vocab, missing, initial_embeddings = self._read_fasttext_embeddings(vocab, embeddings_file)
    self.embeddings = self.dynet_param_collection.param_col.lookup_parameters_from_numpy(initial_embeddings)

    logger.info(f"{in_vocab} vocabulary matches out of {total_embs} total embeddings; "
                f"{missing} vocabulary words without a pretrained embedding out of {self.vocab_size}")

  def _read_fasttext_embeddings(self, vocab, embeddings_file_handle):
    """
    Reads FastText embeddings from a file. Also prints stats about the loaded embeddings for sanity checking.

    :param vocab: a `Vocab` object containing the vocabulary for the experiment
    :param embeddings_file_handle: A file handle on the embeddings file. The embeddings must be in FastText text
     format.
    :return: A tuple of (total number of embeddings read, # embeddings that match vocabulary words, # vocabulary words
     without a matching embedding, embeddings array).
    """
    _, dimension = next(embeddings_file_handle).split()
    if int(dimension) != self.emb_dim:
      raise Exception(f"An embedding size of {self.emb_dim} was specified, but the pretrained embeddings have size {dimension}")

    # Poor man's Glorot initializer for missing embeddings
    bound = np.sqrt(6/(self.vocab_size + self.emb_dim))

    total_embs = 0
    in_vocab = 0
    missing = 0

    embeddings = np.empty((self.vocab_size, self.emb_dim), dtype='float')
    found = np.zeros(self.vocab_size, dtype='bool_')

    for line in embeddings_file_handle:
      total_embs += 1
      word, vals = line.strip().split(' ', 1)
      if word in vocab.w2i:
        in_vocab += 1
        index = vocab.w2i[word]
        embeddings[index] = np.fromstring(vals, sep=" ")
        found[index] = True

    for i in range(self.vocab_size):
      if not found[i]:
        missing += 1
        embeddings[i] = np.random.uniform(-bound, bound, self.emb_dim)

    return total_embs, in_vocab, missing, embeddings
