import ast
import io
import six

import numpy as np

from xnmt.serialize.serializable import Serializable
from xnmt.vocab import Vocab

###### Classes representing single inputs

class Input(object):
  """
  A template class to represent all inputs.
  """
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Input subclasses")

  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")

  def get_padded_sent(self, token, pad_len):
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sent, represented as a list of tokens
  """
  def __init__(self, words):
    self.words = words

  def __len__(self):
    return len(self.words)

  def __getitem__(self, key):
    return self.words[key]

  def get_padded_sent(self, token, pad_len):
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([token] * pad_len)
    return self.__class__(new_words)

  def __str__(self):
    return " ".join(six.moves.map(str, self.words))

class SentenceInput(SimpleSentenceInput):
  def __init__(self, words):
    super(SentenceInput, self).__init__(words)
    self.annotation = {}

  def annotate(self, key, value):
    self.annotation[key] = value

  def get_padded_sent(self, token, pad_len):
    sent = super(SentenceInput, self).get_padded_sent(token, pad_len)
    sent.annotation = self.annotation
    return sent

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens
  """
  def __init__(self, nparr):
    self.nparr = nparr

  def __len__(self):
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def __getitem__(self, key):
    return self.nparr.__getitem__(key)

  def get_padded_sent(self, token, pad_len):
    """
    :param token: None (replicate last frame) or 0 (pad zeros)
    :param pad_len:
    """
    if pad_len == 0:
      return self
    if token is None:
      new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:,-1], (self.nparr.shape[0], 1)), (self.nparr.shape[0], pad_len)), axis=1)
    elif token == 0:
      new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
    else:
      raise NotImplementedError(f"currently only support 'None' or '0' as, but got '{token}'")
    return ArrayInput(new_nparr)

  def get_array(self):
    return self.nparr

###### Classes that will read in a file and turn it into an input

class InputReader(object):
  def read_sents(self, filename, filter_ids=None):
    """
    :param filename: data file
    :param filter_ids: only read sentences with these ids (0-indexed)
    :returns: iterator over sentences from filename
    """
    raise RuntimeError("Input readers must implement the read_sents function")

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
    f = io.open(filename, encoding='utf-8')
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
    with io.open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield line
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files,
  with one sent per line.
  """
  yaml_tag = u'!PlainTextReader'
  def __init__(self, vocab=None):
    self.vocab = vocab
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    return six.moves.map(lambda l: SimpleSentenceInput([self.vocab.convert(word) for word in l.strip().split()] + \
                                                      [self.vocab.convert(Vocab.ES_STR)]),
               self.iterate_filtered(filename, filter_ids))

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param("vocab", self.vocab)

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

class SegmentationTextReader(PlainTextReader):
  yaml_tag = u'!SegmentationTextReader'

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    def convert(line, segmentation):
      line = line.strip().split()
      ret = SentenceInput(list(six.moves.map(self.vocab.convert, line)) + [self.vocab.convert(Vocab.ES_STR)])
      ret.annotate("segment", list(six.moves.map(int, segmentation.strip().split())))
      return ret

    if type(filename) != list:
      try:
        filename = ast.literal_eval(filename)
      except:
        print("Reading %s with a PlainTextReader instead..." % filename)
        return super(SegmentationTextReader, self).read_sents(filename)

    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    data = []
    with io.open(filename[0], encoding='utf-8') as char_inp,\
         io.open(filename[1], encoding='utf-8') as seg_inp:
      for sent_count, (char_line, seg_line) in enumerate(zip(char_inp, seg_inp)):
        if filter_ids is None or sent_count in filter_ids:
          data.append(convert(char_line, seg_line))
        if max_id is not None and sent_count > max_id:
          break
    return data

  def count_sents(self, filename):
    return super(SegmentationTextReader, self).count_sents(filename[0])

class ContVecReader(InputReader, Serializable):
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
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,word_ind] if transpose=False
  * sents[sent_id][word_ind,feat_ind] if transpose=True
  """
  yaml_tag = u"!ContVecReader"

  def __init__(self, transpose=False):
    self.transpose = transpose

  def read_sents(self, filename, filter_ids=None):
    npzFile = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      npzKeys = [npzKeys[i] for i in filter_ids]
    for idx, key in enumerate(npzKeys):
      inp = npzFile[key]
      if self.transpose:
        inp = inp.transpose()
      if idx % 1000 == 999:
        print("Read {} lines ({:.2f}%) of {} at {}".format(idx+1, float(idx+1)/len(npzKeys)*100, filename, key))
      yield ArrayInput(inp)
    npzFile.close()

  def count_sents(self, filename):
    npzFile = np.load(filename, mmap_mode="r") # for counting sentences, only read the index
    l = len(npzFile.files)
    npzFile.close()
    return l

class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems)
  """
  yaml_tag = u"!IDReader"

  def read_sents(self, filename, filter_ids=None):
    return map(lambda l: int(l.strip()), self.iterate_filtered(filename, filter_ids))

###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader, trg_reader, src_file, trg_file,
                         batcher=None, sample_sents=None, max_num_sents=None, max_src_len=None, max_trg_len=None):
  '''
  A utility function to read a parallel corpus.

  :returns: A tuple of (src_data, trg_data, src_batches, trg_batches) where *_batches = *_data if batcher=None
  '''
  src_data = []
  trg_data = []
  if sample_sents:
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
    filter_ids = None
    src_len, trg_len = 0, 0
  src_train_iterator = src_reader.read_sents(src_file, filter_ids)
  trg_train_iterator = trg_reader.read_sents(trg_file, filter_ids)
  for src_sent, trg_sent in six.moves.zip_longest(src_train_iterator, trg_train_iterator):
    if src_sent is None or trg_sent is None:
      raise RuntimeError(f"training src sentences don't match trg sentences: {src_len or src_reader.count_sents(src_file)} != {trg_len or trg_reader.count_sents(trg_file)}!")
    if max_num_sents and max_num_sents >= len(src_data):
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
