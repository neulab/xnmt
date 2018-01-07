import numpy as np
import os
import io
import six
import ast
from collections import defaultdict
from xnmt.serializer import Serializable
from xnmt.vocab import *
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
    if pad_len == 0:
      return self
    new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
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
    self.overwrite_serialize_param(u"vocab", self.vocab)

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

###### CorpusParser

class CorpusParser(object):
  """A class that can read in corpora for training and testing"""
  
  def __init__(self):
    """
    After __init__, the vocabularies must be available because they will be needed to
    initialize other components (in particular, embedders). Currently this is done by
    calling _read_train_corpus here, but when a vocab is prespecified the data could also
    be loaded in a lazy fashion (would be useful to avoid loading training data when we
    only want to do inference; TODO)
    """
    pass

  def _read_training_corpus(self, training_corpus):
    """Read in the training corpus"""
    raise RuntimeError("CorpusParsers must implement read_training_corpus to read in the training/dev corpora")


class BilingualCorpusParser(CorpusParser, Serializable):
  """A class that reads in bilingual corpora, consists of two InputReaders"""

  yaml_tag = u"!BilingualCorpusParser"
  def __init__(self, training_corpus, src_reader, trg_reader, max_src_len=None, max_trg_len=None,
               max_num_train_sents=None, max_num_dev_sents=None, sample_train_sents=None,
               lazy_read=False):
    """
    :param src_reader: InputReader for source side
    :param trg_reader: InputReader for target side
    :param max_src_len: filter pairs longer than this on the source side
    :param max_src_len: filter pairs longer than this on the target side
    :param max_num_train_sents: only read the first n training sentences
    :param max_num_dev_sents: only read the first n dev sentences
    :param sample_train_sents: sample n sentences without replacement from the training corpus (should probably be used with a prespecified vocab)
    :param lazy_read: if True we don't read the training corpus upon initialization (requires the input reader vocabs being prespecified)
    """
    self.training_corpus = training_corpus
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.max_num_train_sents = max_num_train_sents
    self.max_num_dev_sents = max_num_dev_sents
    self.sample_train_sents = sample_train_sents
    self.train_src_len, self.train_trg_len = None, None
    self.dev_src_len, self.dev_trg_len = None, None
    self.data_was_read = False
    if max_num_train_sents is not None and sample_train_sents is not None: raise RuntimeError("max_num_train_sents and sample_train_sents are mutually exclusive!")
    if not lazy_read:
      self._read_training_corpus(self.training_corpus)

  def get_training_corpus(self):
    """
    Training corpus should not be accessed directly, but via this method, to support lazy corpus reading
    """
    if not self.data_was_read:
      self._read_training_corpus(self.training_corpus)
    return self.training_corpus
  
  def _read_training_corpus(self, training_corpus):
    training_corpus.train_src_data = []
    training_corpus.train_trg_data = []
    self.train_src_len = self.src_reader.count_sents(training_corpus.train_src)
    self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
    if self.sample_train_sents:
      if self.train_src_len != self.train_trg_len: raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len, self.train_trg_len))
      self.sample_train_sents = int(self.sample_train_sents)
      filter_ids = np.random.choice(self.train_src_len, self.sample_train_sents, replace=False)
    elif self.max_num_train_sents:
      if self.train_src_len != self.train_trg_len: raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len, self.train_trg_len))
      filter_ids = list(range(min(self.max_num_train_sents, self.train_trg_len)))
    else:
      filter_ids = None
    src_train_iterator = self.src_reader.read_sents(training_corpus.train_src, filter_ids)
    trg_train_iterator = self.trg_reader.read_sents(training_corpus.train_trg, filter_ids)
    for src_sent, trg_sent in six.moves.zip_longest(src_train_iterator, trg_train_iterator):
      if src_sent is None or trg_sent is None:
        raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len or self.src_reader.count_sents(training_corpus.train_src), self.train_trg_len or self.trg_reader.count_sents(training_corpus.train_trg)))
      src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
      trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
      if src_len_ok and trg_len_ok:
        training_corpus.train_src_data.append(src_sent)
        training_corpus.train_trg_data.append(trg_sent)

    self.src_reader.freeze()
    self.trg_reader.freeze()

    training_corpus.dev_src_data = []
    training_corpus.dev_trg_data = []
    if self.max_num_dev_sents:
      self.dev_src_len = self.dev_src_len or self.src_reader.count_sents(training_corpus.dev_src)
      self.dev_trg_len = self.dev_trg_len or self.trg_reader.count_sents(training_corpus.dev_trg)
      if self.dev_src_len != self.dev_trg_len: raise RuntimeError("dev src sentences don't match trg sentences: %s != %s!" % (self.dev_src_len, self.dev_trg_len))
      filter_ids = list(range(min(self.max_num_dev_sents, self.dev_src_len)))
    else:
      filter_ids = None

    src_dev_iterator = self.src_reader.read_sents(training_corpus.dev_src, filter_ids)
    trg_dev_iterator = self.trg_reader.read_sents(training_corpus.dev_trg, filter_ids)
    for src_sent, trg_sent in six.moves.zip_longest(src_dev_iterator, trg_dev_iterator):
      if src_sent is None or trg_sent is None:
        raise RuntimeError("dev src sentences don't match target trg: %s != %s!" % (self.src_reader.count_sents(training_corpus.dev_src), self.dev_trg_len), self.trg_reader.count_sents(training_corpus.dev_trg))
      src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
      trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
      if src_len_ok and trg_len_ok:
        training_corpus.dev_src_data.append(src_sent)
        training_corpus.dev_trg_data.append(trg_sent)
        
    self.data_was_read = True

