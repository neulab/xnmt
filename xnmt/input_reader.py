from itertools import zip_longest

import ast

import numpy as np

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py

from xnmt import logger
from xnmt.input import SimpleSentenceInput, ArrayInput
from xnmt.persistence import serializable_init, Serializable
from xnmt.vocab import Vocab


###### Classes that will read in a file and turn it into an input

class InputReader(object):
  def read_sents(self, filename, filter_ids=None):
    """
    :param filename: data file
    :param filter_ids: only read sentences with these ids (0-indexed)
    :returns: iterator over sentences from filename
    """
    if self.vocab is None:
      self.vocab = Vocab()
    return self.iterate_filtered(filename, filter_ids)

  def read_sent(self, sentence, filter_ids=None):
    """
    :param sentence: a single input string
    :param filter_ids: only read sentences with these ids (0-indexed)
    :returns: a SentenceInput object for the input sentence
    """
    raise RuntimeError("Input readers must implement the read_sent function")

  def count_sents(self, filename):
    """
    :param filename: data file
    :returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def freeze(self):
    pass

class BaseTextReader(InputReader):
  def count_sents(self, filename):
    f = open(filename, encoding='utf-8')
    try:
      return sum(1 for _ in f)
    finally:
      f.close()

  def iterate_filtered(self, filename, filter_ids=None):
    """
    :param filename: data file (text file)
    :param filter_ids:
    :returns: iterator over lines as strings (useful for subclasses to implement read_sents)
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
  Handles the typical case of reading plain text files,
  with one sent per line.
  """
  yaml_tag = '!PlainTextReader'
  @serializable_init
  def __init__(self, vocab=None):
    self.vocab = vocab
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sent(self, sentence, filter_ids=None):
    return SimpleSentenceInput([self.vocab.convert(word) for word in sentence.strip().split()] + \
                                                       [self.vocab.convert(Vocab.ES_STR)])

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

class BaseTextReader(InputReader):
  """
  A base class for text-based :class:`xnmt.input_reader.InputReader` subclasses that implements some helper methods.
  """
  def count_sents(self, filename):
    f = open(filename, encoding='utf-8')
    try:
      return sum(1 for _ in f)
    finally:
      f.close()

  def iterate_filtered(self, filename, filter_ids=None):
    """
    Args:
      filename (str): data file (text file)
      filter_ids (list of int):
    Returns:
      iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield line
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

class SegmentationTextReader(PlainTextReader):
  yaml_tag = '!SegmentationTextReader'

  # TODO: document me

  @serializable_init
  def __init__(self, vocab=None):
    super().__init__(vocab=vocab)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    def convert(line, segmentation):
      line = line.strip().split() + [Vocab.ES_STR]
      ret = SimpleSentenceInput([self.vocab.convert(w) for w in line])
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

class TextToBOWReader(InputReader, Serializable):
  yaml_tag = u'!TextToBOWReader'

  def __init__(self, vocab=None):
    self.vocab = vocab
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def count_sents(self, filename):
    with open(self, filename) as fp:
      return len(fp.keys())

  def count_words(self, trg_words):
    trg_cnt = 0
    for x in trg_words:
      if type(x) == int:
        trg_cnt += 1 if x != Vocab.ES else 0
      else:
        trg_cnt += sum([1 if y != Vocab.ES else 0 for y in x])
    return trg_cnt

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    if filter_ids is not None:
      logger.warning("filter_ids is not implemented in TextToBOWReader yet!")
    with open(filename) as fp:
      for line in fp:
        line = [self.vocab.convert(word) for word in line.strip().split()]
        line.append(self.vocab.convert(self.vocab.ES_STR))
        bow = {}
        for word in line:
          bow[word] = bow.get(word, 0) + 1
        yield SimpleSentenceInput(line, {"bow": bow})

class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems).

  Files must be text files containing a single integer per line.
  """
  yaml_tag = "!IDReader"

  @serializable_init
  def __init__(self):
    pass

  def read_sents(self, filename, filter_ids=None):
    return map(lambda l: int(l.strip()), self.iterate_filtered(filename, filter_ids))

###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader, trg_reader, src_file, trg_file,
                         batcher=None, sample_sents=None, max_num_sents=None, max_src_len=None, max_trg_len=None):
  '''
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
  '''
  src_data = []
  trg_data = []
  if sample_sents:
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    if max_num_sents and max_num_sents < src_len: src_len = trg_len = max_num_sents
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
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

  # Pack batches
  if batcher != None:
    src_batches, trg_batches = batcher.pack(src_data, trg_data)
  else:
    src_batches, trg_batches = src_data, trg_data

  return src_data, trg_data, src_batches, trg_batches
