import time
import sys
import os.path
import subprocess
from collections import defaultdict
import unicodedata
import re
import math
from typing import Dict, List, Optional, Sequence, Union
import numbers

import numpy as np
import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py
import yaml

from xnmt import logger
from xnmt.persistence import serializable_init, Serializable
import xnmt.thirdparty.speech_features as speech_features
import xnmt.utils as utils


class PreprocTask(object):
  def run_preproc_task(self, overwrite: bool = False) -> None:
    ...

class PreprocRunner(Serializable):
  """
  Preprocess and filter the input files, and create the vocabulary.

  Args:
    tasks: A list of preprocessing steps, usually parametrized by in_files (the input files), out_files (the output files), and spec for that particular preprocessing type
           The types of arguments that preproc_spec expects:
           * Option("in_files", help_str="list of paths to the input files"),
           * Option("out_files", help_str="list of paths for the output files"),
           * Option("spec", help_str="The specifications describing which type of processing to use. For normalize and vocab, should consist of the 'lang' and 'spec', where 'lang' can either be 'all' to apply the same type of processing to all languages, or a zero-indexed integer indicating which language to process."),
    overwrite: Whether to overwrite files if they already exist.
  """
  yaml_tag = "!PreprocRunner"

  @serializable_init
  def __init__(self, tasks: Optional[List[PreprocTask]] = None, overwrite: bool = False) -> None:
    if tasks is None: tasks = []
    logger.info("> Preprocessing")

    for task in tasks:
      # Sanity check
      if len(task.in_files) != len(task.out_files):
        raise RuntimeError("Length of in_files and out_files in preprocessor must be identical")
      task.run_preproc_task(overwrite = overwrite)

class PreprocExtract(PreprocTask, Serializable):
  yaml_tag = "!PreprocExtract"
  @serializable_init
  def __init__(self, in_files: Sequence[str], out_files: Sequence[str], specs: 'Extractor') -> None:
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs

  def run_preproc_task(self, overwrite: bool = False) -> None:
    extractor = self.specs
    for in_file, out_file in zip(self.in_files, self.out_files):
      if overwrite or not os.path.isfile(out_file):
        utils.make_parent_dir(out_file)
        extractor.extract_to(in_file, out_file)

class PreprocTokenize(PreprocTask, Serializable):
  yaml_tag = "!PreprocTokenize"
  @serializable_init
  def __init__(self, in_files: Sequence[str], out_files: Sequence[str], specs: Sequence['Tokenizer']) -> None:
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs

  def run_preproc_task(self, overwrite: bool = False) -> None:
    tokenizers = {my_opts["filenum"]: [tok
          for tok in my_opts["tokenizers"]]
          for my_opts in self.specs}
    for file_num, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        utils.make_parent_dir(out_file)
        my_tokenizers = tokenizers.get(file_num, tokenizers["all"])
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          for tokenizer in my_tokenizers:
            in_stream = tokenizer.tokenize_stream(in_stream)
          for line in in_stream:
            out_stream.write(f"{line}\n")

class PreprocNormalize(PreprocTask, Serializable):
  yaml_tag = "!PreprocNormalize"
  @serializable_init
  def __init__(self, in_files: Sequence[str], out_files: Sequence[str], specs: Sequence['Normalizer']):
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs

  def run_preproc_task(self, overwrite: bool = False) -> None:
    normalizers = {my_opts["filenum"]: [norm
          for norm in my_opts["normalizers"]]
          for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        utils.make_parent_dir(out_file)
        my_normalizers = normalizers.get(i, normalizers["all"])
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          for line in in_stream:
            line = line.strip()
            for normalizer in my_normalizers:
              line = normalizer.normalize(line)
            out_stream.write(line + "\n")

class PreprocFilter(PreprocTask, Serializable):
  yaml_tag = "!PreprocFilter"
  @serializable_init
  def __init__(self, in_files: Sequence[str], out_files: Sequence[str], specs: Sequence['SentenceFilterer']):
    self.in_files = in_files
    self.out_files = out_files
    self.filters = specs
  def run_preproc_task(self, overwrite=False):
    # TODO: This will only work with plain-text sentences at the moment. It would be nice if it plays well with the readers
    #       in input.py
    out_streams = [open(x, 'w', encoding='utf-8') if overwrite or not os.path.isfile(x) else None for x in self.out_files]
    if any(x is not None for x in out_streams):
      in_streams = [open(x, 'r', encoding='utf-8') for x in self.in_files]
      for in_lines in zip(*in_streams):
        in_lists = [line.strip().split() for line in in_lines]
        if all([my_filter.keep(in_lists) for my_filter in self.filters]):
          for in_line, out_stream in zip(in_lines, out_streams):
            out_stream.write(in_line)
      for x in in_streams:
        x.close()
    for x in out_streams:
      if x is not None:
        x.close()


class PreprocVocab(PreprocTask, Serializable):
  yaml_tag = "!PreprocVocab"
  @serializable_init
  def __init__(self, in_files: Sequence[str], out_files: Sequence[str], specs: Sequence[dict]) -> None:
    self.in_files = in_files
    self.out_files = out_files
    self.specs = specs# TODO: should use YAML style object passing here

  def run_preproc_task(self, overwrite: bool = False) -> None:
    filters = {my_opts["filenum"]: [norm
          for norm in my_opts["filters"]]
          for my_opts in self.specs}
    for i, (in_file, out_file) in enumerate(zip(self.in_files, self.out_files)):
      if overwrite or not os.path.isfile(out_file):
        utils.make_parent_dir(out_file)
        with open(out_file, "w", encoding='utf-8') as out_stream, \
             open(in_file, "r", encoding='utf-8') as in_stream:
          vocab = {}
          for line in in_stream:
            for word in line.strip().split():
              vocab[word] = vocab.get(word, 0) + 1
          for my_filter in filters.get(i, filters["all"]):
            vocab = my_filter.filter(vocab)
          for word in vocab.keys():
            out_stream.write((word + u"\n"))



##### Preprocessors

class Normalizer(object):
  """A type of normalization to perform to a file. It is initialized first, then expanded."""

  def normalize(self, sent: str) -> str:
    """Takes a plain text string and converts it into another plain text string after preprocessing."""
    raise RuntimeError("Subclasses of Normalizer must implement the normalize() function")

class NormalizerLower(Normalizer, Serializable):
  """Lowercase the text."""

  yaml_tag = "!NormalizerLower"

  def normalize(self, sent: str) -> str:
    return sent.lower()

class NormalizerRemovePunct(Normalizer, Serializable):
  """Remove punctuation from the text.

  Args:
    remove_inside_word: If ``False``, only remove punctuation appearing adjacent to white space.
    allowed_chars: Specify punctuation that is allowed and should not be removed.
  """
  yaml_tag = "!NormalizerRemovePunct"

  @serializable_init
  def __init__(self, remove_inside_word: bool = False, allowed_chars: str = "") -> None:
    self.remove_inside_word = remove_inside_word
    self.exclude = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
                                                              and chr(i) not in set(allowed_chars))
  def normalize(self, sent: str) -> str:
    if self.remove_inside_word:
      ret = ''.join(ch for ch in sent if ch not in self.exclude)
    else:
      words = []
      for w in sent.split():
        words.append(w.strip(''.join(ch for ch in self.exclude)))
      ret = " ".join(words)
    return " ".join(ret.split())

###### Tokenizers

class Tokenizer(Normalizer):
  """
  Pass the text through an internal or external tokenizer.

  TODO: only StreamTokenizers are supported by the preproc runner right now.
  """

  def tokenize(self, sent: str) -> str:
    raise RuntimeError("Subclasses of Tokenizer must implement tokenize() or tokenize_stream()")

  def tokenize_stream(self, stream):
    """
    Tokenize a file-like text stream.

    Args:
      stream: A file-like stream of untokenized text
    Returns:
      A file-like stream of tokenized text

    """
    logger.debug("****** calling tokenize_stream {}".format(self.__class__))
    for line in stream:
      yield self.tokenize(line.strip())

class BPETokenizer(Tokenizer, Serializable):
  """
  Class for byte-pair encoding tokenizer.

  TODO: Unimplemented
  """
  yaml_tag = '!BPETokenizer'

  @serializable_init
  def __init__(self, vocab_size, train_files):
    """Determine the BPE based on the vocab size and corpora"""
    raise NotImplementedError("BPETokenizer is not implemented")

  def tokenize(self, sent):
    """Tokenizes a single sentence according to the determined BPE."""
    raise NotImplementedError("BPETokenizer is not implemented")

class CharacterTokenizer(Tokenizer, Serializable):
  """
  Tokenize into characters, with __ indicating blank spaces
  """
  yaml_tag = '!CharacterTokenizer'

  @serializable_init
  def __init__(self) -> None:
    pass

  def tokenize(self, sent: str) -> str:
    """Tokenizes a single sentence into characters."""
    return ' '.join([('__' if x == ' ' else x) for x in sent])

class UnicodeTokenizer(Tokenizer, Serializable):
  """
  Tokenizer that inserts whitespace between words and punctuation.

  This tokenizer is language-agnostic and (optionally) reversible, and is based on unicode character categories.
  See appendix of https://arxiv.org/pdf/1804.08205

  Args:
    use_merge_symbol: whether to prepend a merge-symbol so that the tokenization becomes reversible
    merge_symbol: the merge symbol to use
    reverse: whether to reverse tokenization (assumes use_merge_symbol=True was used in forward direction)
  """
  yaml_tag = '!UnicodeTokenizer'

  @serializable_init
  def __init__(self, use_merge_symbol: bool = True, merge_symbol: str = '↹', reverse: bool = False) -> None:
    self.merge_symbol = merge_symbol if use_merge_symbol else ''
    self.reverse = reverse

  def tokenize(self, sent: str) -> str:
    """Tokenizes a single sentence.

    Args:
      sent: input sentence
    Returns:
      output sentence
    """
    str_list = []
    if not self.reverse:
      for i in range(len(sent)):
        c = sent[i]
        c_p = sent[i - 1] if i > 0 else c
        c_n = sent[i + 1] if i < len(sent) - 1 else c

        if not UnicodeTokenizer._is_weird(c):
          str_list.append(c)
        else:
          if not c_p.isspace():
            str_list.append(' ' + self.merge_symbol)
            str_list.append(c)
          if not c_n.isspace() and not UnicodeTokenizer._is_weird(c_n):
            str_list.append(self.merge_symbol + ' ')
    else: # self.reverse==True
      i = 0
      while i < len(sent):
        c = sent[i]
        c_n = sent[i + 1] if i < len(sent) - 1 else c
        c_nn = sent[i + 2] if i < len(sent) - 2 else c

        if c + c_n == ' ' + self.merge_symbol and UnicodeTokenizer._is_weird(c_nn):
          i += 2
        elif UnicodeTokenizer._is_weird(c) and c_n + c_nn == self.merge_symbol + ' ':
          str_list.append(c)
          i += 3
        else:
          str_list.append(c)
          i += 1
    return ''.join(str_list)

  @staticmethod
  def _is_weird(c):
    return not (unicodedata.category(c)[0] in 'LMN' or c.isspace())

class ExternalTokenizer(Tokenizer, Serializable):
  """
  Class for arbitrary external tokenizer that accepts untokenized text to stdin and
  emits tokenized tezt to stdout, with passable parameters.

  It is assumed that in general, external tokenizers will be more efficient when run
  once per file, so are run as such (instead of one-execution-per-line.)

  Args:
    path
    tokenizer_args
    arg_separator
  """
  yaml_tag = '!ExternalTokenizer'

  @serializable_init
  def __init__(self, path: str, tokenizer_args: Optional[Sequence[str]] = None, arg_separator: str = ' ') -> None:
    """Initialize the wrapper around the external tokenizer. """
    if tokenizer_args is None: tokenizer_args = {}
    tokenizer_options = []
    if arg_separator != ' ':
      tokenizer_options = [option + arg_separator + str(tokenizer_args[option])
          for option in tokenizer_args]
    else:
      for option in tokenizer_args:
        tokenizer_options.extend([option, str(tokenizer_args[option])])
    self.tokenizer_command = [path] + tokenizer_options
    print(self.tokenizer_command)

  def tokenize(self, sent: str) -> str:
    """
    Pass the sentence through the external tokenizer.

    Args:
      sent: An untokenized sentence
    Return:
      A tokenized sentence

    """
    encode_proc = subprocess.Popen(self.tokenizer_command, stdin=subprocess.PIPE
        , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if isinstance(sent, str):
      sent = sent.encode('utf-8')
    stdout, stderr = encode_proc.communicate(sent)
    if isinstance(stdout, bytes):
      stdout = stdout.decode('utf-8')
    if stderr:
      if isinstance(stderr, bytes):
        stderr = stderr.decode('utf-8')
      sys.stderr.write(stderr + '\n')
    return stdout

class SentencepieceTokenizer(Tokenizer, Serializable):
  """
  Sentencepiece tokenizer
  The options supported by the SentencepieceTokenizer are almost exactly those presented in the Sentencepiece `readme <https://github.com/google/sentencepiece/blob/master/README.md>`_, namely:

  Args:
    train_files
    vocab_size: fixes the vocabulary size
    overwrite
    model_prefix: The trained bpe model will be saved under ``{model_prefix}.model``/``.vocab``
    output_format
    model_type: Either ``unigram`` (default), ``bpe``, ``char`` or ``word``.
                Please refer to the sentencepiece documentation for more details
    hard_vocab_limit: setting this to ``False`` will make the vocab size a soft limit.
                      Useful for small datasets. This is ``True`` by default.
    encode_extra_options:
    decode_extra_options:
  """

  yaml_tag = '!SentencepieceTokenizer'

  @serializable_init
  def __init__(self,
               train_files: Sequence[str],
               vocab_size: numbers.Integral,
               overwrite: bool = False,
               model_prefix: str = 'sentpiece',
               output_format: str = 'piece',
               model_type: str  = 'bpe',
               hard_vocab_limit: bool = True,
               encode_extra_options: Optional[str] = None,
               decode_extra_options: Optional[str] = None) -> None:
    """
    This will initialize and train the sentencepiece tokenizer.

    If overwrite is set to False, learned model will not be overwritten, even if parameters
    are changed.

    "File" output for Sentencepiece written to StringIO temporarily before being written to disk.

    """
    self.model_prefix = model_prefix
    self.output_format = output_format
    self.input_format = output_format
    self.overwrite = overwrite
    self.encode_extra_options = ['--extra_options='+encode_extra_options] if encode_extra_options else []
    self.decode_extra_options = ['--extra_options='+decode_extra_options] if decode_extra_options else []

    utils.make_parent_dir(model_prefix)
    self.sentpiece_train_args = ['--input=' + ','.join(train_files),
                                 '--model_prefix=' + str(model_prefix),
                                 '--vocab_size=' + str(vocab_size),
                                 '--hard_vocab_limit=' + str(hard_vocab_limit).lower(),
                                 '--model_type=' + str(model_type)
                                ]

    self.sentpiece_processor = None

  def init_sentencepiece(self) -> None:
    import sentencepiece as spm
    if ((not os.path.exists(self.model_prefix + '.model')) or
        (not os.path.exists(self.model_prefix + '.vocab')) or
        self.overwrite):
      # This calls sentencepiece. It's pretty verbose
      spm.SentencePieceTrainer.Train(' '.join(self.sentpiece_train_args))
    
    self.sentpiece_processor = spm.SentencePieceProcessor()
    self.sentpiece_processor.Load('%s.model' % self.model_prefix)

    self.sentpiece_encode = self.sentpiece_processor.EncodeAsPieces if self.output_format == 'piece' else self.sentpiece_processor.EncodeAsIds

  
  def tokenize(self, sent: str) -> str:
    """Tokenizes a single sentence into pieces."""
    if self.sentpiece_processor is None:
        self.init_sentencepiece()
    return ' '.join(self.sentpiece_encode(sent))


##### Sentence filterers

class SentenceFilterer(object):
  """Filters sentences that don't match a criterion."""

  def __init__(self, spec: dict) -> None:
    """Initialize the filterer from a specification."""
    pass

  def keep(self, sents: list) -> bool:
    """Takes a list of inputs/outputs for a single sentence and decides whether to keep them.

    In general, these inputs/outpus should already be segmented into words, so len() will return the number of words,
    not the number of characters.

    Args:
      sents: A list of parallel sentences.
    Returns:
      True if they should be used or False if they should be filtered.
    """
    raise RuntimeError("Subclasses of SentenceFilterer must implement the keep() function")

class SentenceFiltererMatchingRegex(SentenceFilterer):
  """Filters sentences via regular expressions.
  A sentence must match the expression to be kept.
  """
  def __init__(self,
               regex_src: Optional[str],
               regex_trg: Optional[str],
               regex_all: Optional[Sequence[str]]) -> None:
    """Specifies the regular expressions to filter the sentences that we'll be getting.

    Args:
      regex_src: regular expression for source language (language index 0)
      regex_trg: regular expression for target language (language index 1)
      regex_all: list of regular expressions for all languages in order
    """
    self.regex = {}
    if regex_src: self.regex[0] = regex_src
    if regex_trg: self.regex[1] = regex_trg
    if regex_all:
      for i, v in enumerate(regex_all):
        self.regex[i] = v

  def keep(self, sents: list) -> bool:
    """ Keep only sentences that match the regex.
    """
    for i, sent in enumerate(sents):
      if type(sent) == list:
        sent = " ".join(sent)
      if self.regex.get(i) is not None:
        if re.search(self.regex[i], sent) is None:
          return False
    return True

class SentenceFiltererLength(SentenceFilterer, Serializable):
  """Filters sentences by length"""

  yaml_tag = "!SentenceFiltererLength"

  @serializable_init
  def __init__(self,
               min_src: Optional[numbers.Integral] = None,
               max_src: Optional[numbers.Integral] = None,
               min_trg: Optional[numbers.Integral] = None,
               max_trg: Optional[numbers.Integral] = None,
               min_all: Union[None, numbers.Integral, Sequence[Optional[numbers.Integral]]] = None,
               max_all: Union[None, numbers.Integral, Sequence[Optional[numbers.Integral]]] = None) -> None:
    """Specifies the length limitations on the sentences that we'll be getting.

    Args:
      min_src: min length of src side
      max_src: max length of src side
      min_trg: min length of trg side
      max_trg: max length of trg side
      min_all: min length; can be a single integer for all languages, or a list with individual values per language
      max_all: max length; can be a single integer for all languages, or a list with individual values per language
    """
    self.each_min = {}
    self.each_max = {}
    self.overall_min = min_all if isinstance(min_all, numbers.Integral) else -1
    self.overall_max = max_all if isinstance(max_all, numbers.Integral) else -1
    if min_src is not None: self.each_min[0] = min_src
    if max_src is not None: self.each_max[0] = max_src
    if min_trg is not None: self.each_min[1] = min_trg
    if max_trg is not None: self.each_max[1] = max_trg
    if isinstance(min_all, list):
      for i, v in enumerate(min_all):
        self.each_min[i] = v
    if isinstance(max_all, list):
      for i, v in enumerate(max_all):
        self.each_max[i] = v

  def keep(self, sents: list) -> bool:
    """Filter sentences by length."""
    for i, sent in enumerate(sents):
      if type(sent) == str:
        raise RuntimeError("length-based sentence filterer does not support `str` input at the moment")
      my_max = self.each_max.get(i, self.overall_max)
      my_min = self.each_min.get(i, self.overall_min)
      if len(sent) < my_min or (my_max != -1 and len(sent) > my_max):
        return False
    return True

##### Vocab filterers

class VocabFilterer(object):
  """Filters vocabulary by some criterion"""

  def __init__(self, spec: dict) -> None:
    """Initialize the filterer from a specification."""
    pass

  def filter(self, vocab: Dict[str,numbers.Integral]) -> Dict[str,numbers.Integral]:
    """Filter a vocabulary.

    Args:
      vocab: A dictionary of vocabulary words with their frequecies.
    Returns:
      A new dictionary with frequencies containing only the words to leave in the vocabulary.
    """
    raise RuntimeError("Subclasses of VocabFilterer must implement the filter() function")

class VocabFiltererFreq(VocabFilterer, Serializable):
  """Filter the vocabulary, removing words below a particular minimum frequency"""
  yaml_tag = "!VocabFiltererFreq"
  @serializable_init
  def __init__(self, min_freq: numbers.Integral) -> None:
    """Specification contains a single value min_freq"""
    self.min_freq = min_freq

  def filter(self, vocab):
    return {k: v for k, v in vocab.items() if v >= self.min_freq}

class VocabFiltererRank(VocabFilterer, Serializable):
  """Filter the vocabulary, removing words above a particular frequency rank"""
  yaml_tag = "!VocabFiltererRank"
  @serializable_init
  def __init__(self, max_rank: numbers.Integral) -> None:
    """Specification contains a single value max_rank"""
    self.max_rank = max_rank

  def filter(self, vocab):
    if len(vocab) <= self.max_rank:
      return vocab
    return {k: v for k, v in sorted(vocab.items(), key=lambda x: -x[1])[:self.max_rank]}

##### Preprocessors

class Extractor(object):
  """A type of extraction task to perform."""

  def extract_to(self, in_file: str, out_file: str) -> None:
    raise RuntimeError("Subclasses of Extractor must implement the extract_to() function")

class MelFiltExtractor(Extractor, Serializable):
  yaml_tag = "!MelFiltExtractor"
  @serializable_init
  def __init__(self, nfilt: numbers.Integral = 40, delta: bool = False):
    self.delta = delta
    self.nfilt = nfilt
  def extract_to(self, in_file: str, out_file: str) -> None:
    """
    Args:
      in_file: yaml file that contains a list of dictionaries.
               Each dictionary contains:
               - wav (str): path to wav file
               - offset (float): start time stamp (optional)
               - duration (float): stop time stamp (optional)
               - speaker: speaker id for normalization (optional; if not given, the filename is used as speaker id)
      out_file: a filename ending in ".h5"
    """
    import librosa
    if not out_file.endswith(".h5"): raise ValueError(f"out_file must end in '.h5', was '{out_file}'")
    start_time = time.time()
    with open(in_file) as in_stream, \
         h5py.File(out_file, "w") as hf:
      db = yaml.load(in_stream)
      db_by_speaker = defaultdict(list)
      for db_index, db_item in enumerate(db):
        speaker_id = db_item.get("speaker", db_item["wav"].split("/")[-1])
        db_item["index"] = db_index
        db_by_speaker[speaker_id].append(db_item)
      for speaker_id in db_by_speaker.keys():
        data = []
        for db_item in db_by_speaker[speaker_id]:
          y, sr = librosa.load(db_item["wav"], sr=16000,
                               offset=db_item.get("offset", 0.0),
                               duration=db_item.get("duration", None))
          if len(y)==0: raise ValueError(f"encountered an empty or out of bounds segment: {db_item}")
          logmel = speech_features.logfbank(y, samplerate=sr, nfilt=self.nfilt)
          if self.delta:
            delta = speech_features.calculate_delta(logmel)
            features = np.concatenate([logmel, delta], axis=1)
          else:
            features = logmel
          data.append(features)
        mean, std = speech_features.get_mean_std(np.concatenate(data))
        for features, db_item in zip(data, db_by_speaker[speaker_id]):
          features = speech_features.normalize(features, mean, std)
          hf.create_dataset(str(db_item["index"]), data=features)
    logger.debug(f"feature extraction took {time.time()-start_time:.3f} seconds")

class LatticeFromPlfExtractor(Extractor, Serializable):
  """
  Creates node-labeled lattices that can be read by the ``LatticeInputReader``.

  The input to this extractor is a list of edge-labeled lattices in PLF format. The PLF format is described here:
  http://www.statmt.org/moses/?n=Moses.WordLattices
  It is used, among others, in the Fisher/Callhome Spanish-to-English Speech Translation Corpus (Post et al, 2013).
  """

  yaml_tag = "!LatticeFromPlfExtractor"

  def extract_to(self, in_file: str, out_file: str):

    output_file = open(out_file, "w")

    counter, num_node_sum1, num_edge_sum1, num_node_sum2, num_edge_sum2 = 0, 0, 0, 0, 0
    with open(in_file) as f:
      for line in f:
        graph = LatticeFromPlfExtractor._Lattice()
        graph.read_plf_line(line)
        graph.insert_initial_node()
        graph.insert_final_node()
        graph.forward()
        graph2 = LatticeFromPlfExtractor._Lattice.convert_to_node_labeled_lattice(graph)
        if len(graph2.nodes) == 1:
          graph2.insert_initial_node()
        serial = graph2.serialize_to_string()
        output_file.write(serial + "\n")
        counter += 1
        num_node_sum1 += len(graph.nodes)
        num_node_sum2 += len(graph2.nodes)
        num_edge_sum1 += len(graph.edges)
        num_edge_sum2 += len(graph2.edges)
        if counter % 1000 == 0:
          logger.info(f"finished {counter} lattices.")

    output_file.close()

    logger.info(f"avg # nodes, # edges for edge-labeled lattices: {float(num_node_sum1) / counter}, {float(num_edge_sum1) / counter}")
    logger.info(f"avg # nodes, # edges for node-labeled lattices: {float(num_node_sum2) / counter}, {float(num_edge_sum2) / counter}")

  class _Lattice(object):

    def __init__(self, nodes=None, edges=None):
      self.nodes = nodes
      self.edges = edges

    def serialize_to_string(self):
      node_ids = {}
      for n in self.nodes:
        node_ids[n] = len(node_ids)
      node_lst = [n.label for n in self.nodes]
      node_str = "[" + ", ".join(["(" + ", ".join(
        [("'" + str(labelItem) + "'" if type(labelItem) == str else str(labelItem)) for labelItem in node]) + ")" for
                                 node in node_lst]) + "]"
      numbered_edges = []
      for fromNode, toNode, _ in self.edges:
        from_id = node_ids[fromNode]
        to_id = node_ids[toNode]
        numbered_edges.append((from_id, to_id))
      edge_str = str(numbered_edges)
      return node_str + "," + edge_str

    def insert_initial_node(self):
      initial_node = LatticeFromPlfExtractor._LatticeLabel(label=("<s>", 0.0, 0.0, 0.0))
      if len(self.nodes) > 0:
        self.edges.insert(0, (initial_node, self.nodes[0], LatticeFromPlfExtractor._LatticeLabel(("<s>", 0.0))))
      self.nodes.insert(0, initial_node)

    def insert_final_node(self):
      final_node = LatticeFromPlfExtractor._LatticeLabel(label=("final-node", 0.0))
      self.edges.append((self.nodes[-1], final_node, LatticeFromPlfExtractor._LatticeLabel(("</s>", 0.0))))
      self.nodes.append(final_node)

    @staticmethod
    def convert_to_node_labeled_lattice(edge_labeled_lattice):
      word_nodes = []
      word_node_edges = []
      for edge in edge_labeled_lattice.edges:
        _, _, edge_label = edge
        new_word_node = edge_label
        word_nodes.append(new_word_node)
      for edge1 in edge_labeled_lattice.edges:
        _, edge1_to, edge1_label = edge1
        for edge2 in edge_labeled_lattice.edges:
          edge2_from, _, edge2_label = edge2
          if edge1_to == edge2_from:
            word_node_edges.append((edge1_label, edge2_label, LatticeFromPlfExtractor._LatticeLabel()))
      return LatticeFromPlfExtractor._Lattice(nodes=word_nodes, edges=word_node_edges)

    def forward(self):
      """
      Use the forward algorithm to add marginal link probs and backward-normalized lattice weights to the graph
      """
      self.nodes[0].marginal_log_prob = 0.0
      for edge in self.edges:
        from_node, to_node, edge_label = edge
        prev_sum = 0.0  # incomplete P(toNode)
        if not hasattr(to_node, 'marginal_log_prob'):
          to_node.marginal_log_prob = 0.0
        else:
          prev_sum = math.exp(to_node.marginal_log_prob)
        fwd_weight = math.exp(edge_label.label[1])  # lattice weight normalized across outgoing edges
        marginal_link_prob = math.exp(from_node.marginal_log_prob) * fwd_weight  # P(fromNode, toNode)
        to_node.marginal_log_prob = math.log(prev_sum + marginal_link_prob)  # (partially) completed P(toNode)
        to_node.label = (to_node.marginal_log_prob,)
        edge_label.label = tuple(list(edge_label.label) + [min(0.0, math.log(marginal_link_prob))])
      for node in self.nodes:
        incoming_edges = [edge for edge in self.edges if edge[1] == node]
        incoming_sum = sum([math.exp(edge[0].marginal_log_prob) for edge in incoming_edges])
        for edge in incoming_edges:
          from_node, to_node, edge_label = edge
          bwd_weight_log = min(0.0, edge[0].marginal_log_prob - math.log(incoming_sum))
          edge_label.label = tuple(list(edge_label.label) + [bwd_weight_log])

    def read_plf_line(self, line):
      parenth_depth = 0
      plf_nodes = []
      plf_edges = []

      for token in re.split("([()])", line):
        if len(token.strip()) > 0 and token.strip() != ",":
          if token == "(":
            parenth_depth += 1
            if parenth_depth == 2:
              new_node = LatticeFromPlfExtractor._LatticeLabel(label=None)
              plf_nodes.append(new_node)
          elif token == ")":
            parenth_depth -= 1
            if parenth_depth == 0:
              new_node = LatticeFromPlfExtractor._LatticeLabel(label=None)
              plf_nodes.append(new_node)
              break  # end of the lattice
          elif token[0] == "'":
            word, score, distance = [eval(tt) for tt in token.split(",")]
            cur_node_id = len(plf_nodes) - 1
            edge_from = cur_node_id
            edge_to = cur_node_id + distance
            edge_label = LatticeFromPlfExtractor._LatticeLabel(label=(word, score))
            plf_edges.append((edge_from, edge_to, edge_label))
      resolved_edges = []
      for edge in plf_edges:
        edge_from, edge_to, edge_label = edge
        resolved_edges.append((plf_nodes[edge_from], plf_nodes[edge_to], edge_label))
      self.nodes = plf_nodes
      self.edges = resolved_edges

  class _LatticeLabel(object):
    def __init__(self, label=None):
      self.label = label
    def __repr__(self):
      return str(self.label)
