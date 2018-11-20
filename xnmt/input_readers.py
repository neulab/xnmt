import ast
from itertools import zip_longest
from functools import lru_cache
import ast
from typing import Any, Iterator, Optional, Sequence, Union
import numbers

import numpy as np

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py

from xnmt import logger

from xnmt import batchers, output, sent, events, vocabs
from xnmt.persistence import serializable_init, Serializable

class InputReader(object):
  """
  A base class to read in a file and turn it into an input
  """
  def read_sents(self, filename: str, filter_ids: Sequence[numbers.Integral] = None) -> Iterator[sent.Sentence]:
    """
    Read sentences and return an iterator.

    Args:
      filename: data file
      filter_ids: only read sentences with these ids (0-indexed)
    Returns: iterator over sentences from filename
    """
    return self.iterate_filtered(filename, filter_ids)

  def count_sents(self, filename: str) -> int:
    """
    Count the number of sentences in a data file.

    Args:
      filename: data file
    Returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def needs_reload(self) -> bool:
    """
    Overwrite this method if data needs to be reload for each epoch
    """
    return False

class BaseTextReader(InputReader):

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.Sentence:
    """
    Convert a raw text line into an input object.

    Args:
      line: a single input string
      idx: sentence number
    Returns: a SentenceInput object for the input sentence
    """
    raise RuntimeError("Input readers must implement the read_sent function")

  @lru_cache(maxsize=128)
  def count_sents(self, filename: str) -> numbers.Integral:
    newlines = 0
    with open(filename, 'r+b') as f:
      for _ in f:
        newlines += 1
    return newlines

  def iterate_filtered(self, filename: str, filter_ids: Optional[Sequence[numbers.Integral]]=None) -> Iterator:
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
          yield self.read_sent(line=line, idx=sent_count)
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

def convert_int(x: Any) -> numbers.Integral:
  try:
    return int(x)
  except ValueError:
    raise ValueError(f"Expecting integer tokens because no vocab was set. Got: '{x}'")

class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files, with one sent per line.

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
    read_sent_len: if set, read the length of each sentence instead of the sentence itself. EOS is not counted.
    output_proc: output processors to revert the created sentences back to a readable string
  """
  yaml_tag = '!PlainTextReader'
 
  @serializable_init
  def __init__(self,
               vocab: Optional[vocabs.Vocab] = None,
               read_sent_len: bool = False,
               output_proc: Sequence[output.OutputProcessor] = []) -> None:
    self.vocab = vocab
    self.read_sent_len = read_sent_len
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.Sentence:
    if self.vocab:
      convert_fct = self.vocab.convert
    else:
      convert_fct = convert_int
    if self.read_sent_len:
      return sent.ScalarSentence(idx=idx, value=len(line.strip().split()))
    else:
      return sent.SimpleSentence(idx=idx,
                                 words=[convert_fct(word) for word in line.strip().split()] + [vocabs.Vocab.ES],
                                 vocab=self.vocab,
                                 output_procs=self.output_procs)

  def vocab_size(self) -> numbers.Integral:
    return len(self.vocab)

class CompoundReader(InputReader, Serializable):
  """
  A compound reader reads inputs using several input readers at the same time.

  The resulting inputs will be of type :class:`sent.CompoundSentence`, which holds the results from the different
  readers as a tuple. Inputs can be read from different locations (if input file name is a sequence of filenames) or all
  from the same location (if it is a string). The latter can be used to read the same inputs using several input
  different readers which might capture different aspects of the input data.

  Args:
    readers: list of input readers to use
    vocab: not used by this reader, but some parent components may require access to the vocab.
  """
  yaml_tag = "!CompoundReader"
  @serializable_init
  def __init__(self, readers: Sequence[InputReader], vocab: Optional[vocabs.Vocab] = None) -> None:
    if len(readers) < 2: raise ValueError("need at least two readers")
    self.readers = readers
    if vocab: self.vocab = vocab

  def read_sents(self, filename: Union[str,Sequence[str]], filter_ids: Sequence[numbers.Integral] = None) \
          -> Iterator[sent.Sentence]:
    if isinstance(filename, str): filename = [filename] * len(self.readers)
    generators = [reader.read_sents(filename=cur_filename, filter_ids=filter_ids) for (reader, cur_filename) in
                     zip(self.readers, filename)]
    while True:
      try:
        sub_sents = tuple([next(gen) for gen in generators])
        yield sent.CompoundSentence(sents=sub_sents)
      except StopIteration:
        return

  def count_sents(self, filename: str) -> int:
    return self.readers[0].count_sents(filename if isinstance(filename,str) else filename[0])

  def needs_reload(self) -> bool:
    return any(reader.needs_reload() for reader in self.readers)


class SentencePieceTextReader(BaseTextReader, Serializable):
  """
  Read in text and segment it with sentencepiece. Optionally perform sampling
  for subword regularization, only at training time.
  https://arxiv.org/pdf/1804.10959.pdf 
  """
  yaml_tag = '!SentencePieceTextReader'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               model_file: str,
               sample_train: bool=False,
               l: numbers.Integral=-1,
               alpha: numbers.Real=0.1,
               vocab: Optional[vocabs.Vocab]=None,
               output_proc=[output.JoinPieceTextOutputProcessor]) -> None:
    """
    Args:
      model_file: The sentence piece model file
      sample_train: On the training set, sample outputs
      l: The "l" parameter for subword regularization, how many sentences to sample
      alpha: The "alpha" parameter for subword regularization, how much to smooth the distribution
      vocab: The vocabulary
      output_proc: output processors to revert the created sentences back to a readable string
    """
    import sentencepiece as spm
    self.subword_model = spm.SentencePieceProcessor()
    self.subword_model.Load(model_file)
    self.sample_train = sample_train
    self.l = l
    self.alpha = alpha
    self.vocab = vocab
    self.train = False
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.SimpleSentence:
    if self.sample_train and self.train:
      words = self.subword_model.SampleEncodeAsPieces(line.strip(), self.l, self.alpha)
    else:
      words = self.subword_model.EncodeAsPieces(line.strip())
    words = [w.decode('utf-8') for w in words]
    return sent.SimpleSentence(idx=idx,
                               words=[self.vocab.convert(word) for word in words] + [self.vocab.convert(vocabs.Vocab.ES_STR)],
                               vocab=self.vocab,
                               output_procs=self.output_procs)

  def vocab_size(self) -> numbers.Integral:
    return len(self.vocab)

class RamlTextReader(BaseTextReader, Serializable):
  """
  Handles the RAML sampling, can be used on the target side, or on both the source and target side.
  Randomly replaces words according to Hamming Distance.
  https://arxiv.org/pdf/1808.07512.pdf
  https://arxiv.org/pdf/1609.00150.pdf
  """
  yaml_tag = '!RamlTextReader'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               tau: Optional[float] = 1.,
               vocab: Optional[vocabs.Vocab] = None,
               output_proc: Sequence[output.OutputProcessor]=[]) -> None:
    """
    Args:
      tau: The temperature that controls peakiness of the sampling distribution
      vocab: The vocabulary
    """
    self.tau = tau
    self.vocab = vocab
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.SimpleSentence:
    words = line.strip().split()
    if not self.train:
      return sent.SimpleSentence(idx=idx,
                                 words=[self.vocab.convert(word) for word in words] + [vocabs.Vocab.ES],
                                 vocab=self.vocab,
                                 output_procs=self.output_procs)
    word_ids = np.array([self.vocab.convert(word) for word in words])
    length = len(word_ids)
    logits = np.arange(length) * (-1) * self.tau
    logits = np.exp(logits - np.max(logits))
    probs = logits / np.sum(logits)
    num_words = np.random.choice(length, p=probs)
    corrupt_pos = np.random.binomial(1, p=num_words/length, size=(length,))
    num_words_to_sample = np.sum(corrupt_pos)
    sampled_words = np.random.choice(np.arange(2, len(self.vocab)), size=(num_words_to_sample,))
    word_ids[np.where(corrupt_pos==1)[0].tolist()] = sampled_words
    return sent.SimpleSentence(idx=idx,
                               words=word_ids.tolist() + [vocabs.Vocab.ES],
                               vocab=self.vocab,
                               output_procs=self.output_procs)

  def needs_reload(self) -> bool:
    return True 

class CharFromWordTextReader(PlainTextReader, Serializable):
  """
  Read in word based corpus and turned that into SegmentedSentence.
  SegmentedSentece's words are characters, but it contains the information of the segmentation.
  
  x = SegmentedSentence("i code today")
  (TRUE) x.words == ["i", "c", "o", "d", "e", "t", "o", "d", "a", "y"]
  (TRUE) x.segment == [0, 4, 9]

  It means that the segmentation (end of words) happen in the 0th, 4th and 9th position of the char sequence.
  """
  yaml_tag = "!CharFromWordTextReader"
  @serializable_init
  def __init__(self,
               vocab: vocabs.Vocab = None,
               read_sent_len: bool = False,
               output_proc: Sequence[output.OutputProcessor] = []) -> None:
    self.vocab = vocab
    self.read_sent_len = read_sent_len
    self.output_procs = output.OutputProcessor.get_output_processor(output_proc)

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.SegmentedSentence:
    chars = []
    segs = []
    offset = 0
    for word in line.strip().split():
      offset += len(word)
      segs.append(offset-1)
      chars.extend([c for c in word])
    segs.append(len(chars))
    chars.append(vocabs.Vocab.ES_STR)
    sent_input = sent.SegmentedSentence(segment=segs,
                                        words=[self.vocab.convert(c) for c in chars],
                                        idx=idx,
                                        vocab=self.vocab,
                                        output_procs=self.output_procs)
    return sent_input

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
    transpose: whether inputs are transposed or not.
    feat_from: use feature dimensions in a range, starting at this index (inclusive)
    feat_to: use feature dimensions in a range, ending at this index (exclusive)
    feat_skip: stride over features
    timestep_skip: stride over timesteps
    timestep_truncate: cut off timesteps if sequence is longer than specified value
  """
  yaml_tag = u"!H5Reader"
  @serializable_init
  def __init__(self,
               transpose: bool = False,
               feat_from: Optional[numbers.Integral] = None,
               feat_to: Optional[numbers.Integral] = None,
               feat_skip: Optional[numbers.Integral] = None,
               timestep_skip: Optional[numbers.Integral] = None,
               timestep_truncate: Optional[numbers.Integral] = None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename: str, filter_ids: Optional[Sequence[numbers.Integral]]=None) -> Iterator[sent.ArraySentence]:
    with h5py.File(filename, "r") as hf:
      h5_keys = sorted(hf.keys(), key=lambda x: int(x))
      if filter_ids is not None:
        filter_ids = sorted(filter_ids)
        h5_keys = [h5_keys[i] for i in filter_ids]
        h5_keys.sort(key=lambda x: int(x))
      for sent_no, key in enumerate(h5_keys):
        inp = hf[key][:]
        if self.transpose:
          inp = inp.transpose()

        sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
        if sub_inp.size < inp.size:
          inp = np.empty_like(sub_inp)
          np.copyto(inp, sub_inp)
        else:
          inp = sub_inp

        if sent_no % 1000 == 999:
          logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(h5_keys)*100:.2f}%) of {filename} at {key}")
        yield sent.ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)

  def count_sents(self, filename: str) -> numbers.Integral:
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
    transpose: whether inputs are transposed or not.
    feat_from: use feature dimensions in a range, starting at this index (inclusive)
    feat_to: use feature dimensions in a range, ending at this index (exclusive)
    feat_skip: stride over features
    timestep_skip: stride over timesteps
    timestep_truncate: cut off timesteps if sequence is longer than specified value
  """
  yaml_tag = u"!NpzReader"
  @serializable_init
  def __init__(self,
               transpose: bool = False,
               feat_from: Optional[numbers.Integral] = None,
               feat_to: Optional[numbers.Integral] = None,
               feat_skip: Optional[numbers.Integral] = None,
               timestep_skip: Optional[numbers.Integral] = None,
               timestep_truncate: Optional[numbers.Integral] = None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename: str, filter_ids: Optional[Sequence[numbers.Integral]] = None) -> None:
    npzFile = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      filter_ids = sorted(filter_ids)
      npzKeys = [npzKeys[i] for i in filter_ids]
      npzKeys.sort(key=lambda x: int(x.split('_')[-1]))
    for sent_no, key in enumerate(npzKeys):
      inp = npzFile[key]
      if self.transpose:
        inp = inp.transpose()

      sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
      if sub_inp.size < inp.size:
        inp = np.empty_like(sub_inp)
        np.copyto(inp, sub_inp)
      else:
        inp = sub_inp

      if sent_no % 1000 == 999:
        logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(npzKeys)*100:.2f}%) of {filename} at {key}")
      yield sent.ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)
    npzFile.close()

  def count_sents(self, filename: str) -> numbers.Integral:
    npz_file = np.load(filename, mmap_mode="r")  # for counting sentences, only read the index
    l = len(npz_file.files)
    npz_file.close()
    return l

class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems).

  Files must be text files containing a single integer per line.
  """
  yaml_tag = "!IDReader"

  @serializable_init
  def __init__(self) -> None:
    pass

  def read_sent(self, line: str, idx: numbers.Integral) -> sent.ScalarSentence:
    return sent.ScalarSentence(idx=idx, value=int(line.strip()))

  def read_sents(self, filename: str, filter_ids: Optional[Sequence[numbers.Integral]] = None) -> list:
    return [l for l in self.iterate_filtered(filename, filter_ids)]


class LatticeReader(BaseTextReader, Serializable):
  """
  Reads lattices from a text file.

  The expected lattice file format is as follows:
  * 1 line per lattice
  * lines are serialized python lists / tuples
  * 2 lists per lattice:
    - list of nodes, with every node a 4-tuple: (lexicon_entry, fwd_log_prob, marginal_log_prob, bwd_log_prob)
    - list of arcs, each arc a tuple: (node_id_start, node_id_end)
            - node_id references the nodes and is 0-indexed
            - node_id_start < node_id_end
  * All paths must share a common start and end node, i.e. <s> and </s> need to be contained in the lattice

  A simple example lattice:
    [('<s>', 0.0, 0.0, 0.0), ('buenas', 0, 0.0, 0.0), ('tardes', 0, 0.0, 0.0), ('</s>', 0.0, 0.0, 0.0)],[(0, 1), (1, 2), (2, 3)]

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
    text_input: If ``True``, assume a standard text file as input and convert it to a flat lattice.
    flatten: If ``True``, convert to a flat lattice, with all probabilities set to 1.
  """
  yaml_tag = '!LatticeReader'

  @serializable_init
  def __init__(self, vocab:vocabs.Vocab, text_input: bool = False, flatten = False):
    self.vocab = vocab
    self.text_input = text_input
    self.flatten = flatten

  def read_sent(self, line, idx):
    if self.text_input:
      nodes = [sent.LatticeNode(nodes_prev=[], nodes_next=[1], value=vocabs.Vocab.SS,
                                fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0)]
      for word in line.strip().split():
        nodes.append(
          sent.LatticeNode(nodes_prev=[len(nodes)-1], nodes_next=[len(nodes)+1], value=self.vocab.convert(word),
                           fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0))
      nodes.append(
        sent.LatticeNode(nodes_prev=[len(nodes) - 1], nodes_next=[], value=vocabs.Vocab.ES,
                         fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0))

    else:
      node_list, arc_list = ast.literal_eval(line)
      nodes = [sent.LatticeNode(nodes_prev=[], nodes_next=[],
                                value=self.vocab.convert(item[0]),
                                fwd_log_prob=item[1], marginal_log_prob=item[2], bwd_log_prob=item[3])
               for item in node_list]
      if self.flatten:
        for node_i in range(len(nodes)):
          if node_i < len(nodes)-1: nodes[node_i].nodes_next.append(node_i+1)
          if node_i > 0: nodes[node_i].nodes_prev.append(node_i-1)
          nodes[node_i].fwd_log_prob = nodes[node_i].bwd_log_prob = nodes[node_i].marginal_log_prob = 0.0
      else:
        for from_index, to_index in arc_list:
          nodes[from_index].nodes_next.append(to_index)
          nodes[to_index].nodes_prev.append(from_index)

      assert nodes[0].value == self.vocab.SS
      assert nodes[-1].value == self.vocab.ES

    return sent.Lattice(idx=idx, nodes=nodes, vocab=self.vocab)

  def vocab_size(self):
    return len(self.vocab)


###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader: InputReader,
                         trg_reader: InputReader,
                         src_file: str,
                         trg_file: str,
                         batcher: batchers.Batcher=None,
                         sample_sents: Optional[numbers.Integral] = None,
                         max_num_sents: Optional[numbers.Integral] = None,
                         max_src_len: Optional[numbers.Integral] = None,
                         max_trg_len: Optional[numbers.Integral] = None) -> tuple:
  """
  A utility function to read a parallel corpus.

  Args:
    src_reader:
    trg_reader:
    src_file:
    trg_file:
    batcher:
    sample_sents: if not None, denote the number of sents that should be randomly chosen from all available sents.
    max_num_sents: if not None, read only the first this many sents
    max_src_len: skip pair if src side is too long
    max_trg_len: skip pair if trg side is too long

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
    src_len_ok = max_src_len is None or src_sent.sent_len() <= max_src_len
    trg_len_ok = max_trg_len is None or trg_sent.sent_len() <= max_trg_len
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
