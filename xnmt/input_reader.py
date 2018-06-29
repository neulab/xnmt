from itertools import zip_longest
from functools import lru_cache
import ast
from typing import Sequence, Iterator

import numpy as np

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py

from xnmt import logger
from xnmt.input import SimpleSentenceInput, AnnotatedSentenceInput, ArrayInput
from xnmt.persistence import serializable_init, Serializable
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.vocab import Vocab
import xnmt.input

class InputReader(object):
  """
  A base class to read in a file and turn it into an input
  """
  def read_sents(self, filename: str, filter_ids: Sequence[int] = None) -> Iterator[xnmt.input.Input]:
    """
    Read sentences and return an iterator.

    Args:
      filename: data file
      filter_ids: only read sentences with these ids (0-indexed)
    Returns: iterator over sentences from filename
    """
    if self.vocab is None:
      self.vocab = Vocab()
    return self.iterate_filtered(filename, filter_ids)

  def count_sents(self, filename: str) -> int:
    """
    Count the number of sentences in a data file.

    Args:
      filename: data file
    Returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def freeze(self) -> None:
    """
    Freeze the data representation, e.g. by freezing the vocab.
    """
    pass

  def needs_reload(self) -> bool:
    """
    Overwrite this method if data needs to be reload for each epoch
    """
    return False

class BaseTextReader(InputReader):

  def read_sent(self, line: str) -> xnmt.input.Input:
    """
    Convert a raw text line into an input object.

    Args:
      line: a single input string
    Returns: a SentenceInput object for the input sentence
    """
    raise RuntimeError("Input readers must implement the read_sent function")

  @lru_cache(maxsize=128)
  def count_sents(self, filename):
    newlines = 0
    f = open(filename, 'r+b')
    for line in f:
      newlines += 1
    return newlines

  def iterate_filtered(self, filename, filter_ids=None):
    """
    Args:
      filename: data file (text file)
      filter_ids:
    Returns: iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield self.read_sent(line)
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files, with one sent per line.
  """
  yaml_tag = '!PlainTextReader'

  @serializable_init
  def __init__(self, vocab=None, include_vocab_reference=False):
    self.vocab = vocab
    self.include_vocab_reference = include_vocab_reference
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sent(self, line):
    vocab_reference = self.vocab if self.include_vocab_reference else None
    return SimpleSentenceInput([self.vocab.convert(word) for word in line.strip().split()] + \
                                                       [self.vocab.convert(Vocab.ES_STR)], vocab_reference)

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.save_processed_arg("vocab", self.vocab)

  def vocab_size(self):
    return len(self.vocab)

class SentencePieceTextReader(BaseTextReader, Serializable):
  """
  Read in text and segment it with sentencepiece. Optionally perform sampling
  for subword regularization, only at training time.
  https://arxiv.org/pdf/1804.10959.pdf 
  """
  yaml_tag = '!SentencePieceTextReader'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, model_file, sample_train=False, l=-1, alpha=0.1, vocab=None, include_vocab_reference=False):
    """
    Args:
      model_file: The sentence piece model file
      sample_train: On the training set, sample outputs
      l: The "l" parameter for subword regularization, how many sentences to sample
      alpha: The "alpha" parameter for subword regularization, how much to smooth the distribution
      vocab: The vocabulary
      include_vocab_reference: Whether to include the vocab with the input
    """
    import sentencepiece as spm
    self.subword_model = spm.SentencePieceProcessor()
    self.subword_model.Load(model_file)
    self.sample_train = sample_train
    self.l = l
    self.alpha = alpha
    self.vocab = vocab
    self.include_vocab_reference = include_vocab_reference
    self.train = False
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def read_sent(self, sentence):
    vocab_reference = self.vocab if self.include_vocab_reference else None
    if self.sample_train and self.train:
      words = self.subword_model.SampleEncodeAsPieces(sentence.strip(), self.l, self.alpha)
    else:
      words = self.subword_model.EncodeAsPieces(sentence.strip())
    return SimpleSentenceInput([self.vocab.convert(word) for word in words] + \
                                                       [self.vocab.convert(Vocab.ES_STR)], vocab_reference)

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.save_processed_arg("vocab", self.vocab)

  def count_words(self, trg_words):
    trg_cnt = 0
    for x in trg_words:
      if type(x) == int:
        trg_cnt += 1 if x != Vocab.ES else 0
      else:
        trg_cnt += sum([1 if y != Vocab.ES else 0 for y in x])
    return trg_cnt

  def vocab_size(self):
    return len(self.vocab)

  def needs_reload(self):
    return True

class SegmentationTextReader(PlainTextReader):
  yaml_tag = '!SegmentationTextReader'
  
  # TODO: document me

  @serializable_init
  def __init__(self, vocab=None, include_vocab_reference=False):
    super().__init__(vocab=vocab, include_vocab_reference=include_vocab_reference)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    def convert(line, segmentation):
      line = line.strip().split()
      ret = AnnotatedSentenceInput(list(map(self.vocab.convert, line)) + [self.vocab.convert(Vocab.ES_STR)])
      ret.annotate("segment", list(map(int, segmentation.strip().split())))
      return ret

    if type(filename) != list:
      try:
        filename = ast.literal_eval(filename)
      except:
        logger.debug("Reading %s with a PlainTextReader instead..." % filename)
        return super(SegmentationTextReader, self).read_sents(filename)

    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    data = []
    with open(filename[0], encoding='utf-8') as char_inp,\
         open(filename[1], encoding='utf-8') as seg_inp:
      for sent_count, (char_line, seg_line) in enumerate(zip(char_inp, seg_inp)):
        if filter_ids is None or sent_count in filter_ids:
          data.append(convert(char_line, seg_line))
        if max_id is not None and sent_count > max_id:
          break
    return data

  def count_sents(self, filename):
    return super(SegmentationTextReader, self).count_sents(filename[0])


class H5Reader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".h5" file, which can be created for example using xnmt.preproc.MelFiltExtractor

  The data items are assumed to be labeled with integers 0, 1, .. (converted to strings).

  Each data item will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,timestep] if transpose=False
  * sents[sent_id][timestep,feat_ind] if transpose=True

  Args:
    transpose (bool): whether inputs are transposed or not.
    feat_from (int): use feature dimensions in a range, starting at this index (inclusive)
    feat_to (int): use feature dimensions in a range, ending at this index (exclusive)
    feat_skip (int): stride over features
    timestep_skip (int): stride over timesteps
    timestep_truncate (int): cut off timesteps if sequence is longer than specified value
  """
  yaml_tag = u"!H5Reader"
  @serializable_init
  def __init__(self, transpose=False, feat_from=None, feat_to=None, feat_skip=None, timestep_skip=None,
               timestep_truncate=None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename, filter_ids=None):
    with h5py.File(filename, "r") as hf:
      h5_keys = sorted(hf.keys(), key=lambda x: int(x))
      if filter_ids is not None:
        h5_keys = [h5_keys[i] for i in filter_ids]
        h5_keys.sort(key=lambda x: int(x))
      for idx, key in enumerate(h5_keys):
        inp = hf[key][:]
        if self.transpose:
          inp = inp.transpose()

        sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
        if sub_inp.size < inp.size:
          inp = np.empty_like(sub_inp)
          np.copyto(inp, sub_inp)
        else:
          inp = sub_inp

        if idx % 1000 == 999:
          logger.info(f"Read {idx+1} lines ({float(idx+1)/len(h5_keys)*100:.2f}%) of {filename} at {key}")
        yield ArrayInput(inp)

  def count_sents(self, filename):
    with h5py.File(filename, "r") as hf:
      l = len(hf.keys())
    return l


class NpzReader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".npz" file, which consists of multiply ".npy" files, each
  corresponding to a single sequence of continuous features. This can be
  created in two ways:
  * Use the builtin function numpy.savez_compressed()
  * Create a bunch of .npy files, and run "zip" on them to zip them into an archive.

  The file names should be named XXX_0, XXX_1, etc., where the final number after the underbar
  indicates the order of the sequence in the corpus. This is done automatically by
  numpy.savez_compressed(), in which case the names will be arr_0, arr_1, etc.

  Each numpy file will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable.
  * sents[sent_id][feat_ind,timestep] if transpose=False
  * sents[sent_id][timestep,feat_ind] if transpose=True

  Args:
    transpose (bool): whether inputs are transposed or not.
    feat_from (int): use feature dimensions in a range, starting at this index (inclusive)
    feat_to (int): use feature dimensions in a range, ending at this index (exclusive)
    feat_skip (int): stride over features
    timestep_skip (int): stride over timesteps
    timestep_truncate (int): cut off timesteps if sequence is longer than specified value
  """
  yaml_tag = u"!NpzReader"
  @serializable_init
  def __init__(self, transpose=False, feat_from=None, feat_to=None, feat_skip=None, timestep_skip=None,
               timestep_truncate=None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename, filter_ids=None):
    npzFile = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      npzKeys = [npzKeys[i] for i in filter_ids]
      npzKeys.sort(key=lambda x: int(x.split('_')[-1]))
    for idx, key in enumerate(npzKeys):
      inp = npzFile[key]
      if self.transpose:
        inp = inp.transpose()

      sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
      if sub_inp.size < inp.size:
        inp = np.empty_like(sub_inp)
        np.copyto(inp, sub_inp)
      else:
        inp = sub_inp

      if idx % 1000 == 999:
        logger.info(f"Read {idx+1} lines ({float(idx+1)/len(npzKeys)*100:.2f}%) of {filename} at {key}")
      yield ArrayInput(inp)
    npzFile.close()

  def count_sents(self, filename):
    npzFile = np.load(filename, mmap_mode="r")  # for counting sentences, only read the index
    l = len(npzFile.files)
    npzFile.close()
    return l


class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems).
  
  Files must be text files containing a single integer per line.
  """
  yaml_tag = "!IDReader"

  @serializable_init
  def __init__(self):
    pass

  def read_sent(self, line):
    return xnmt.input.IntInput(int(line.strip()))

  def read_sents(self, filename, filter_ids=None):
    return [l for l in self.iterate_filtered(filename, filter_ids)]

###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader, trg_reader, src_file, trg_file,
                         batcher=None, sample_sents=None, max_num_sents=None, max_src_len=None, max_trg_len=None):
  """
  A utility function to read a parallel corpus.

  Args:
    src_reader (InputReader):
    trg_reader (InputReader):
    src_file (str):
    trg_file (str):
    batcher (Batcher):
    sample_sents (int): if not None, denote the number of sents that should be randomly chosen from all available sents.
    max_num_sents (int): if not None, read only the first this many sents
    max_src_len (int): skip pair if src side is too long
    max_trg_len (int): skip pair if trg side is too long

  Returns:
    A tuple of (src_data, trg_data, src_batches, trg_batches) where ``*_batches = *_data`` if ``batcher=None``
  """
  src_data = []
  trg_data = []
  if sample_sents:
    logger.info(f"Starting to read {sample_sents} parallel sentences of {src_file} and {trg_file}")
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    if max_num_sents and max_num_sents < src_len: src_len = trg_len = max_num_sents
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
    logger.info(f"Starting to read {src_file} and {trg_file}")
    filter_ids = None
    src_len, trg_len = 0, 0
  src_train_iterator = src_reader.read_sents(src_file, filter_ids)
  trg_train_iterator = trg_reader.read_sents(trg_file, filter_ids)
  for src_sent, trg_sent in zip_longest(src_train_iterator, trg_train_iterator):
    if src_sent is None or trg_sent is None:
      raise RuntimeError(f"training src sentences don't match trg sentences: {src_len or src_reader.count_sents(src_file)} != {trg_len or trg_reader.count_sents(trg_file)}!")
    if max_num_sents and (max_num_sents <= len(src_data)):
      break
    src_len_ok = max_src_len is None or len(src_sent) <= max_src_len
    trg_len_ok = max_trg_len is None or len(trg_sent) <= max_trg_len
    if src_len_ok and trg_len_ok:
      src_data.append(src_sent)
      trg_data.append(trg_sent)

  logger.info(f"Done reading {src_file} and {trg_file}. Packing into batches.")

  # Pack batches
  if batcher is not None:
    src_batches, trg_batches = batcher.pack(src_data, trg_data)
  else:
    src_batches, trg_batches = src_data, trg_data

  logger.info(f"Done packing batches.")

  return src_data, trg_data, src_batches, trg_batches
