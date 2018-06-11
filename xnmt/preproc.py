import time
import sys
import os.path
import subprocess
from collections import defaultdict
import unicodedata
import re

import numpy as np
import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py
import yaml

from xnmt import logger
from xnmt.persistence import serializable_init, Serializable
from xnmt.speech_features import logfbank, calculate_delta, get_mean_std, normalize
from xnmt.util import make_parent_dir

##### Preprocessors

class Normalizer(object):
  """A type of normalization to perform to a file. It is initialized first, then expanded."""

  def __init__(self, spec=None):
    """Initialize the normalizer from a specification."""
    pass

  def normalize(self, sent):
    """Takes a plain text string and converts it into another plain text string after preprocessing."""
    raise RuntimeError("Subclasses of Normalizer must implement the normalize() function")

  @staticmethod
  def from_spec(spec):
    """Takes a list of normalizer specifications, and returns the appropriate processors."""
    preproc_list = []
    if spec is not None:
      for my_spec in spec:
        if my_spec["type"] == "lower":
          preproc_list.append(NormalizerLower(my_spec))
        elif my_spec["type"] == "remove_punct":
          preproc_list.append(NormalizerRemovePunct(my_spec))
        else:
          raise RuntimeError("Unknown normalizer type {}".format(my_spec["type"]))
    return preproc_list

class NormalizerLower(Normalizer):
  """Lowercase the text."""

  def normalize(self, sent):
    return sent.lower()

class NormalizerRemovePunct(Normalizer):
  """Remove punctuation from the text."""
  def __init__(self, spec=None):
    self.remove_inside_word = spec.get("remove_inside_word", False)
    self.exclude = set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
                                                              and chr(i) not in set(spec.get("allowed_chars", "")))
  def normalize(self, sent):
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

  def tokenize(self, sent):
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
  def __init__(self):
    pass

  def tokenize(self, sent):
    """Tokenizes a single sentence into characters."""
    return ' '.join([('__' if x == ' ' else x) for x in sent])

class ExternalTokenizer(Tokenizer, Serializable):
  """
  Class for arbitrary external tokenizer that accepts untokenized text to stdin and
  emits tokenized tezt to stdout, with passable parameters.

  It is assumed that in general, external tokenizers will be more efficient when run
  once per file, so are run as such (instead of one-execution-per-line.)

  """
  yaml_tag = '!ExternalTokenizer'

  @serializable_init
  def __init__(self, path, tokenizer_args=None, arg_separator=' '):
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

  def tokenize(self, sent):
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

    - ``model_type``: Either ``unigram`` (default), ``bpe``, ``char`` or ``word``.
      Please refer to the sentencepiece documentation for more details
    - ``model_prefix``: The trained bpe model will be saved under ``{model_prefix}.model``/``.vocab``
    - ``vocab_size``: fixes the vocabulary size
    - ``hard_vocab_limit``: setting this to ``False`` will make the vocab size a soft limit. 
      Useful for small datasets. This is ``True`` by default.
  """

  yaml_tag = '!SentencepieceTokenizer'

  @serializable_init
  def __init__(self, path, train_files, vocab_size, overwrite=False, model_prefix='sentpiece'
      , output_format='piece', model_type='bpe', hard_vocab_limit=True
      , encode_extra_options=None, decode_extra_options=None):
    """
    This will initialize and train the sentencepiece tokenizer.

    If overwrite is set to False, learned model will not be overwritten, even if parameters
    are changed.

    "File" output for Sentencepiece written to StringIO temporarily before being written to disk.

    """
    # TODO: deprecate the path argument
    self.sentpiece_path = path
    self.model_prefix = model_prefix
    self.output_format = output_format
    self.input_format = output_format
    self.overwrite = overwrite
    self.encode_extra_options = ['--extra_options='+encode_extra_options] if encode_extra_options else []
    self.decode_extra_options = ['--extra_options='+decode_extra_options] if decode_extra_options else []

    make_parent_dir(model_prefix)
    self.sentpiece_train_args = ['--input=' + ','.join(train_files),
                                 '--model_prefix=' + str(model_prefix),
                                 '--vocab_size=' + str(vocab_size),
                                 '--hard_vocab_limit=' + str(hard_vocab_limit).lower(),
                                 '--model_type=' + str(model_type)
                                ]

    self.sentpiece_processor = None

  def init_sentencepiece(self):
    import sentencepiece as spm
    if ((not os.path.exists(self.model_prefix + '.model')) or
        (not os.path.exists(self.model_prefix + '.vocab')) or
        self.overwrite):
      # This calls sentencepiece. It's pretty verbose
      spm.SentencePieceTrainer.Train(' '.join(self.sentpiece_train_args))
    
    self.sentpiece_processor = spm.SentencePieceProcessor()
    self.sentpiece_processor.Load('%s.model' % self.model_prefix)

    self.sentpiece_encode = self.sentpiece_processor.EncodeAsPieces if self.output_format == 'piece' else self.sentpiece_processor.EncodeAsIds

  
  def tokenize(self, sent):
    """Tokenizes a single sentence into pieces."""
    if self.sentpiece_processor is None:
        self.init_sentencepiece()
    return ' '.join(self.sentpiece_encode(sent))


##### Sentence filterers

class SentenceFilterer(object):
  """Filters sentences that don't match a criterion."""

  def __init__(self, spec):
    """Initialize the filterer from a specification."""
    pass

  def keep(self, sents):
    """Takes a list of inputs/outputs for a single sentence and decides whether to keep them.

    In general, these inputs/outpus should already be segmented into words, so len() will return the number of words,
    not the number of characters.

    Args:
      sents: A list of parallel sentences.
    Returns:
      True if they should be used or False if they should be filtered.
    """
    raise RuntimeError("Subclasses of SentenceFilterer must implement the keep() function")

  @staticmethod
  def from_spec(spec):
    """Takes a list of preprocessor specifications, and returns the appropriate processors."""
    preproc_list = []
    if spec is not None:
      for my_spec in spec:
        if my_spec["type"] == "length":
          preproc_list.append(SentenceFiltererLength(my_spec))
        elif my_spec["type"] == "matching-regex":
          preproc_list.append(SentenceFiltererMatchingRegex(my_spec))
        else:
          raise RuntimeError("Unknown preprocessing type {}".format(my_spec["type"]))
    return preproc_list

class SentenceFiltererMatchingRegex(SentenceFilterer):
  """Filters sentences via regular expressions.
  A sentence must match the expression to be kept.
  """

  def __init__(self, spec):
    """Specifies the regular expressions to filter the sentences that we'll be getting.

    The regular expressions are passed as a dictionary with keys as follows:
      regex_INT: This will specify the regular expression for a specific language (zero indexed)
      regex_src: Equivalent to regex_0
      regex_trg: Equivalent to regex_1
    """
    self.regex = {}
    idx_map = {"src": 0, "trg": 1}
    for k, v in spec.items():
      if k == "type":
        pass
      elif k.startswith("regex"):
        _, idx = k.split("_")
        idx_tmp = idx_map.get(idx)
        if idx_tmp is None:
          idx_tmp = int(idx)
        idx = idx_tmp
        self.regex[idx] = v

  def keep(self, sents):
    """ Keep only sentences that match the regex.
    """
    for i, sent in enumerate(sents):
      if type(sent) == list:
        sent = " ".join(sent)
      if self.regex.get(i) is not None:
        if re.search(self.regex[i], sent) is None:
          return False
    return True

class SentenceFiltererLength(SentenceFilterer):
  """Filters sentences by length"""

  def __init__(self, spec):
    """Specifies the type of length limitations on the sentences that we'll be getting.

    The limitations are passed as a dictionary with keys as follows:
      max/min: This will specify the default maximum and minimum length.
      max_INT/min_INT: This will specify limitations for a specific language (zero indexed)
      max_src/min_src: Equivalent to max_0/min_0
      max_trg/min_trg: Equivalent to max_1/min_1
    """
    self.each_max = {}
    self.each_min = {}
    self.overall_max = -1
    self.overall_min = -1
    idx_map = {"src": 0, "trg": 1}
    for k, v in spec.items():
      if k == "type":
        pass
      elif k == "max":
        self.overall_max = v
      elif k == "min":
        self.overall_min = v
      else:
        direc, idx = k.split('_')
        idx_tmp = idx_map.get(idx)
        if idx_tmp is None:
          idx_tmp = int(idx)
        idx = idx_tmp

        if direc == "max":
          self.each_max[idx] = v
        elif direc == "min":
          self.each_min[idx] = v
        else:
          raise RuntimeError("Unknown limitation type {} in length-based sentence filterer".format(k))

  def keep(self, sents):
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

  def __init__(self, spec):
    """Initialize the filterer from a specification."""
    pass

  def filter(self, vocab):
    """Filter a vocabulary.

    Args:
      vocab: A dictionary of vocabulary words with their frequecies.
    Returns:
      A new dictionary with frequencies containing only the words to leave in the vocabulary.
    """
    raise RuntimeError("Subclasses of VocabFilterer must implement the filter() function")

  @staticmethod
  def from_spec(spec):
    """Takes a list of preprocessor specifications, and returns the appropriate processors."""
    preproc_list = []
    if spec is not None:
      for my_spec in spec:
        if my_spec["type"] == "freq":
          preproc_list.append(VocabFiltererFreq(my_spec))
        elif my_spec["type"] == "rank":
          preproc_list.append(VocabFiltererRank(my_spec))
        else:
          raise RuntimeError("Unknown VocabFilterer type {}".format(my_spec["type"]))
    return preproc_list

class VocabFiltererFreq(VocabFilterer):
  """Filter the vocabulary, removing words below a particular minimum frequency"""

  def __init__(self, spec):
    """Specification contains a single value min_freq"""
    self.min_freq = spec["min_freq"]

  def filter(self, vocab):
    return {k: v for k, v in vocab.items() if v >= self.min_freq}

class VocabFiltererRank(VocabFilterer):
  """Filter the vocabulary, removing words above a particular frequency rank"""

  def __init__(self, spec):
    """Specification contains a single value max_rank"""
    self.max_rank = spec["max_rank"]

  def filter(self, vocab):
    if len(vocab) <= self.max_rank:
      return vocab
    return {k: v for k, v in sorted(vocab.items(), key=lambda x: -x[1])[:self.max_rank]}

##### Preprocessors

class Extractor(object):
  """A type of feature extraction to perform."""

  def extract_to(self, in_file, out_file):
    raise RuntimeError("Subclasses of Extractor must implement the extract_to() function")

class MelFiltExtractor(Extractor, Serializable):
  yaml_tag = "!MelFiltExtractor"
  @serializable_init
  def __init__(self, nfilt=40, delta=False):
    self.delta = delta
    self.nfilt = nfilt
  def extract_to(self, in_file, out_file):
    """
    in_file: yaml file that contains a list of dictionaries.
             Each dictionary contains:
             - wav (str): path to wav file
             - offset (float): start time stamp (optional)
             - duration (float
             ): stop time stamp (optional)
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
          logmel = logfbank(y, samplerate=sr, nfilt=self.nfilt)
          if self.delta:
            delta = calculate_delta(logmel)
            features = np.concatenate([logmel, delta], axis=1)
          else:
            features = logmel
          data.append(features)
        mean, std = get_mean_std(np.concatenate(data))
        for features, db_item in zip(data, db_by_speaker[speaker_id]):
          features = normalize(features, mean, std)
          hf.create_dataset(str(db_item["index"]), data=features)
    logger.debug(f"feature extraction took {time.time()-start_time:.3f} seconds")
