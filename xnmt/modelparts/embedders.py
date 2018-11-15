import numbers
from typing import Any, Optional, Union
import io

import numpy as np
import dynet as dy

from xnmt import logger
from xnmt import batchers, events, expression_seqs, input_readers, param_collections, param_initializers, sent, vocabs
from xnmt.modelparts import transforms
from xnmt.persistence import bare, Path, Ref, Serializable, serializable_init

class Embedder(object):
  """
  An embedder takes in word IDs and outputs continuous vectors.

  This can be done on a word-by-word basis, or over a sequence.
  """

  def embed(self, word: Any) -> dy.Expression:
    """Embed a single word.

    Args:
      word: This will generally be an integer word ID, but could also be something like a string. It could
            also be batched, in which case the input will be a :class:`xnmt.batcher.Batch` of integers or other things.

    Returns:
      Expression corresponding to the embedding of the word(s).
    """
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

  def embed_sent(self, x: Any) -> expression_seqs.ExpressionSequence:
    """Embed a full sentence worth of words. By default, just do a for loop.

    Args:
      x: This will generally be a list of word IDs, but could also be a list of strings or some other format.
         It could also be batched, in which case it will be a (possibly masked) :class:`xnmt.batcher.Batch` object

    Returns:
      An expression sequence representing vectors of each word in the input.
    """
    # single mode
    if not batchers.is_batched(x):
      embeddings = [self.embed(word) for word in x]
    # minibatch mode
    else:
      embeddings = []
      seq_len = x.sent_len()
      for single_sent in x: assert single_sent.sent_len()==seq_len
      for word_i in range(seq_len):
        batch = batchers.mark_as_batch([single_sent[word_i] for single_sent in x])
        embeddings.append(self.embed(batch))

    return expression_seqs.ExpressionSequence(expr_list=embeddings, mask=x.mask if batchers.is_batched(x) else None)

  def choose_vocab(self,
                   vocab: vocabs.Vocab,
                   yaml_path: Path,
                   src_reader: input_readers.InputReader,
                   trg_reader: input_readers.InputReader) -> vocabs.Vocab:
    """Choose the vocab for the embedder basd on the passed arguments

    This is done in order of priority of vocab, model+yaml_path

    Args:
      vocab: If None, try to obtain from ``src_reader`` or ``trg_reader``, depending on the ``yaml_path``
      yaml_path: Path of this embedder in the component hierarchy. Automatically determined when deserializing the YAML model.
      src_reader: Model's src_reader, if exists and unambiguous.
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab
    """
    if vocab is not None:
      return len(vocab)
    elif "src_embedder" in yaml_path:
      if src_reader is None or src_reader.vocab is None:
        raise ValueError("Could not determine src_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "trg_embedder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or trg_reader.vocab is None:
        raise ValueError("Could not determine trg_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError("Attempted to determine vocab size of {} (path: {}), but path was not src_embedder, trg_embedder, or output_projector, so it could not determine what part of the model to use. Please set vocab_size or vocab explicitly.".format(self.__class__, yaml_path))

  def choose_vocab_size(self,
                        vocab_size: numbers.Integral,
                        vocab: vocabs.Vocab,
                        yaml_path: Path,
                        src_reader: input_readers.InputReader,
                        trg_reader: input_readers.InputReader) -> int:
    """Choose the vocab size for the embedder basd on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path

    Args:
      vocab_size : vocab size or None
      vocab: vocab or None
      yaml_path: Path of this embedder in the component hierarchy. Automatically determined when YAML-deserializing.
      src_reader: Model's src_reader, if exists and unambiguous.
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab size
    """
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif "src_embedder" in yaml_path:
      if src_reader is None or getattr(src_reader,"vocab",None) is None:
        raise ValueError("Could not determine src_embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "trg_embedder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or trg_reader.vocab is None:
        raise ValueError("Could not determine target embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError(f"Attempted to determine vocab size of {self.__class__} (path: {yaml_path}), "
                       f"but path was not src_embedder, trg_embedder, or output_projector, so it could not determine what part of the model to use. "
                       f"Please set vocab_size or vocab explicitly.")

class DenseWordEmbedder(Embedder, transforms.Linear, Serializable):
  """
  Word embeddings via full matrix.

  Args:
    emb_dim: embedding dimension
    weight_noise: apply Gaussian noise with given standard deviation to embeddings
    word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
    vocab_size: vocab size or None
    vocab: vocab or None
    yaml_path: Path of this embedder in the component hierarchy. Automatically set by the YAML deserializer.
    src_reader: A reader for the source side. Automatically set by the YAML deserializer.
    trg_reader: A reader for the target side. Automatically set by the YAML deserializer.
  """
  yaml_tag = "!DenseWordEmbedder"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               emb_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               weight_noise: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               word_dropout: numbers.Real = 0.0,
               fix_norm: Optional[numbers.Real] = None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init",
                                                                    default=bare(param_initializers.ZeroInitializer)),
               vocab_size: Optional[numbers.Integral] = None,
               vocab: Optional[vocabs.Vocab] = None,
               yaml_path: Path = '',
               src_reader: Optional[input_readers.InputReader] = Ref("model.src_reader", default=None),
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None)) -> None:
    self.fix_norm = fix_norm
    self.weight_noise = weight_noise
    self.word_dropout = word_dropout
    self.emb_dim = emb_dim
    param_collection = param_collections.ParamManager.my_params(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.save_processed_arg("vocab_size", self.vocab_size)
    self.embeddings = param_collection.add_parameters((self.vocab_size, self.emb_dim), init=param_init.initializer((self.vocab_size, self.emb_dim), is_lookup=True))
    self.bias = param_collection.add_parameters((self.vocab_size,), init=bias_init.initializer((self.vocab_size,)))

  @events.handle_xnmt_event
  def on_start_sent(self, *args, **kwargs) -> None:
    self.word_id_mask = None

  @events.handle_xnmt_event
  def on_set_train(self, val: bool) -> None:
    self.train = val

  def embed(self, x: Union[batchers.Batch, numbers.Integral]) -> dy.Expression:
    if self.train and self.word_dropout > 0.0 and self.word_id_mask is None:
      batch_size = x.batch_size() if batchers.is_batched(x) else 1
      self.word_id_mask = [set(np.random.choice(self.vocab_size, int(self.vocab_size * self.word_dropout), replace=False)) for _ in range(batch_size)]
    emb_e = dy.parameter(self.embeddings)
    # single mode
    if not batchers.is_batched(x):
      if self.train and self.word_id_mask and x in self.word_id_mask[0]:
        ret = dy.zeros((self.emb_dim,))
      else:
        ret = dy.pick(emb_e, index=x)
        if self.fix_norm is not None:
          ret = dy.cdiv(ret, dy.l2_norm(ret))
          if self.fix_norm != 1:
            ret *= self.fix_norm
    # minibatch mode
    else:
      ret = dy.pick_batch(emb_e, x)
      if self.fix_norm is not None:
        ret = dy.cdiv(ret, dy.l2_norm(ret))
        if self.fix_norm != 1:
          ret *= self.fix_norm
      if self.train and self.word_id_mask and any(x[i] in self.word_id_mask[i] for i in range(x.batch_size())):
        dropout_mask = dy.inputTensor(np.transpose([[0.0]*self.emb_dim if x[i] in self.word_id_mask[i] else [1.0]*self.emb_dim for i in range(x.batch_size())]), batched=True)
        ret = dy.cmult(ret, dropout_mask)
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    W1 = dy.parameter(self.embeddings)
    b1 = dy.parameter(self.bias)
    return dy.affine_transform([b1, W1, input_expr])


class SimpleWordEmbedder(Embedder, Serializable):
  """
  Simple word embeddings via lookup.

  Args:
    emb_dim: embedding dimension
    weight_noise: apply Gaussian noise with given standard deviation to embeddings
    word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
    param_init: how to initialize lookup matrices
    vocab_size: vocab size or None
    vocab: vocab or None
    yaml_path: Path of this embedder in the component hierarchy. Automatically set by the YAML deserializer.
    src_reader: A reader for the source side. Automatically set by the YAML deserializer.
    trg_reader: A reader for the target side. Automatically set by the YAML deserializer.
  """

  yaml_tag = '!SimpleWordEmbedder'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               emb_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               weight_noise: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               word_dropout: numbers.Real = 0.0,
               fix_norm: Optional[numbers.Real] = None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               vocab_size: Optional[numbers.Integral] = None,
               vocab: Optional[vocabs.Vocab] = None,
               yaml_path: Path = Path(),
               src_reader: Optional[input_readers.InputReader] = Ref("model.src_reader", default=None),
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None)) -> None:
    self.emb_dim = emb_dim
    self.weight_noise = weight_noise
    self.word_dropout = word_dropout
    self.fix_norm = fix_norm
    self.word_id_mask = None
    self.train = False
    param_collection = param_collections.ParamManager.my_params(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.save_processed_arg("vocab_size", self.vocab_size)
    self.embeddings = param_collection.add_lookup_parameters((self.vocab_size, self.emb_dim),
                             init=param_init.initializer((self.vocab_size, self.emb_dim), is_lookup=True))

  @events.handle_xnmt_event
  def on_set_train(self, val: bool) -> None:
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, *args, **kwargs) -> None:
    self.word_id_mask = None

  def embed(self, x: Union[numbers.Integral, batchers.Batch]) -> dy.Expression:
    if self.train and self.word_dropout > 0.0 and self.word_id_mask is None:
      batch_size = x.batch_size() if batchers.is_batched(x) else 1
      self.word_id_mask = [set(np.random.choice(self.vocab_size, int(self.vocab_size * self.word_dropout), replace=False)) for _ in range(batch_size)]
    # single mode
    if not batchers.is_batched(x):
      if self.train and self.word_id_mask and x in self.word_id_mask[0]:
        ret = dy.zeros((self.emb_dim,))
      else:
        ret = self.embeddings[x]
        if self.fix_norm is not None:
          ret = dy.cdiv(ret, dy.l2_norm(ret))
          if self.fix_norm != 1:
            ret *= self.fix_norm
    # minibatch mode
    else:
      ret = self.embeddings.batch(x)
      if self.fix_norm is not None:
        ret = dy.cdiv(ret, dy.l2_norm(ret))
        if self.fix_norm != 1:
          ret *= self.fix_norm
      if self.train and self.word_id_mask and any(x[i] in self.word_id_mask[i] for i in range(x.batch_size())):
        dropout_mask = dy.inputTensor(np.transpose([[0.0]*self.emb_dim if x[i] in self.word_id_mask[i] else [1.0]*self.emb_dim for i in range(x.batch_size())]), batched=True)
        ret = dy.cmult(ret, dropout_mask)
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

class NoopEmbedder(Embedder, Serializable):
  """
  This embedder performs no lookups but only passes through the inputs.

  Normally, the input is a Sentence object, which is converted to an expression.

  Args:
    emb_dim: Size of the inputs
  """

  yaml_tag = '!NoopEmbedder'

  @serializable_init
  def __init__(self, emb_dim: Optional[numbers.Integral]) -> None:
    self.emb_dim = emb_dim

  def embed(self, x: Union[np.ndarray, list]) -> dy.Expression:
    return dy.inputTensor(x, batched=batchers.is_batched(x))

  def embed_sent(self, x: sent.Sentence) -> expression_seqs.ExpressionSequence:
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    batched = batchers.is_batched(x)
    first_sent = x[0] if batched else x
    if hasattr(first_sent, "get_array"):
      if not batched:
        return expression_seqs.LazyNumpyExpressionSequence(lazy_data=x.get_array())
      else:
        return expression_seqs.LazyNumpyExpressionSequence(lazy_data=batchers.mark_as_batch(
                                           [s for s in x]),
                                           mask=x.mask)
    else:
      if not batched:
        embeddings = [self.embed(word) for word in x]
      else:
        embeddings = []
        for word_i in range(x.sent_len()):
          embeddings.append(self.embed(batchers.mark_as_batch([single_sent[word_i] for single_sent in x])))
      return expression_seqs.ExpressionSequence(expr_list=embeddings, mask=x.mask)


class PretrainedSimpleWordEmbedder(SimpleWordEmbedder, Serializable):
  """
  Simple word embeddings via lookup. Initial pretrained embeddings must be supplied in FastText text format.

  Args:
    filename: Filename for the pretrained embeddings
    emb_dim: embedding dimension; if None, use exp_global.default_layer_dim
    weight_noise: apply Gaussian noise with given standard deviation to embeddings; if ``None``, use exp_global.weight_noise
    word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level, see https://arxiv.org/abs/1512.05287
    fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
    vocab: vocab or None
    yaml_path: Path of this embedder in the component hierarchy. Automatically set by the YAML deserializer.
    src_reader: A reader for the source side. Automatically set by the YAML deserializer.
    trg_reader: A reader for the target side. Automatically set by the YAML deserializer.
"""

  yaml_tag = '!PretrainedSimpleWordEmbedder'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               filename: str,
               emb_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               weight_noise: numbers.Real = Ref("exp_global.weight_noise", default=0.0),
               word_dropout: numbers.Real = 0.0,
               fix_norm: Optional[numbers.Real] = None,
               vocab: Optional[vocabs.Vocab] = None,
               yaml_path: Path = Path(),
               src_reader: Optional[input_readers.InputReader] = Ref("model.src_reader", default=None),
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None)) -> None:
    self.emb_dim = emb_dim
    self.weight_noise = weight_noise
    self.word_dropout = word_dropout
    self.word_id_mask = None
    self.train = False
    self.fix_norm = fix_norm
    self.pretrained_filename = filename
    param_collection = param_collections.ParamManager.my_params(self)
    self.vocab = self.choose_vocab(vocab, yaml_path, src_reader, trg_reader)
    self.vocab_size = len(vocab)
    self.save_processed_arg("vocab", self.vocab)
    with open(self.pretrained_filename, encoding='utf-8') as embeddings_file:
      total_embs, in_vocab, missing, initial_embeddings = self._read_fasttext_embeddings(vocab, embeddings_file)
    self.embeddings = param_collection.lookup_parameters_from_numpy(initial_embeddings)

    logger.info(f"{in_vocab} vocabulary matches out of {total_embs} total embeddings; "
                f"{missing} vocabulary words without a pretrained embedding out of {self.vocab_size}")

  def _read_fasttext_embeddings(self, vocab: vocabs.Vocab, embeddings_file_handle: io.IOBase) -> tuple:
    """
    Reads FastText embeddings from a file. Also prints stats about the loaded embeddings for sanity checking.

    Args:
      vocab: a `Vocab` object containing the vocabulary for the experiment
      embeddings_file_handle: A file handle on the embeddings file. The embeddings must be in FastText text
                              format.
    Returns:
      tuple: A tuple of (total number of embeddings read, # embeddings that match vocabulary words, # vocabulary words
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


class PositionEmbedder(Embedder, Serializable):

  yaml_tag = '!PositionEmbedder'

  @serializable_init
  def __init__(self,
               max_pos: numbers.Integral,
               emb_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init",
                                                                     default=bare(param_initializers.GlorotInitializer))) \
          -> None:
    """
    max_pos: largest embedded position
    emb_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.emb_dim = emb_dim
    param_collection = param_collections.ParamManager.my_params(self)
    param_init = param_init
    dim = (self.emb_dim, max_pos)
    self.embeddings = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def embed(self, word): raise NotImplementedError("Position-embedding for individual words not implemented yet.")
  def embed_sent(self, sent_len: numbers.Integral) -> expression_seqs.ExpressionSequence:
    embeddings = dy.strided_select(dy.parameter(self.embeddings), [1,1], [0,0], [self.emb_dim, sent_len])
    return expression_seqs.ExpressionSequence(expr_tensor=embeddings, mask=None)
