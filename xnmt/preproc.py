import logging
logger = logging.getLogger('xnmt')
import time
import sys
import os.path
import subprocess
from collections import defaultdict

import numpy as np
import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py
import yaml

from xnmt.serialize.serializable import Serializable
from xnmt.serialize.serializer import serializable_init
from xnmt.speech_features import logfbank, calculate_delta, get_mean_std, normalize

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
    if spec != None:
      for my_spec in spec:
        if my_spec["type"] == "lower":
          preproc_list.append(NormalizerLower(my_spec))
        else:
          raise RuntimeError("Unknown normalizer type {}".format(my_spec["type"]))
    return preproc_list

class NormalizerLower(Normalizer):
  """Lowercase the text."""

  def normalize(self, sent):
    return sent.lower()

###### Tokenizers

class Tokenizer(Normalizer, Serializable):
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

class BPETokenizer(Tokenizer):
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

class CharacterTokenizer(Tokenizer):
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

class ExternalTokenizer(Tokenizer):
  """
  Class for arbitrary external tokenizer that accepts untokenized text to stdin and
  emits tokenized tezt to stdout, with passable parameters.

  It is assumed that in general, external tokenizers will be more efficient when run
  once per file, so are run as such (instead of one-execution-per-line.)

  """
  yaml_tag = '!ExternalTokenizer'

  @serializable_init
  def __init__(self, path, tokenizer_args={}, arg_separator=' '):
    """Initialize the wrapper around the external tokenizer. """
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
      string = sent.encode('utf-8')
    stdout, stderr = encode_proc.communicate(string)
    if stderr:
      if isinstance(stderr, bytes):
        stderr = stderr.decode('utf-8')
      sys.stderr.write(stderr + '\n')
    return stdout

class SentencepieceTokenizer(ExternalTokenizer):
  """
  A wrapper around an independent installation of the sentencepiece tokenizer
  with passable parameters.
  """
  yaml_tag = '!SentencepieceTokenizer'

  @serializable_init
  def __init__(self, path, train_files, vocab_size, overwrite=False, model_prefix='sentpiece'
      , output_format='piece', model_type='bpe'
      , encode_extra_options=None, decode_extra_options=None):
    """
    Initialize the wrapper around sentencepiece and train the tokenizer.

    If overwrite is set to False, learned model will not be overwritten, even if parameters
    are changed.

    "File" output for Sentencepiece written to StringIO temporarily before being written to disk.

    """
    self.sentpiece_path = path
    self.model_prefix = model_prefix
    self.output_format = output_format
    self.input_format = output_format
    self.encode_extra_options = ['--extra_options='+encode_extra_options] if encode_extra_options else []
    self.decode_extra_options = ['--extra_options='+decode_extra_options] if decode_extra_options else []

    if not os.path.exists(os.path.dirname(model_prefix)):
      try:
        os.makedirs(os.path.dirname(model_prefix))
      except OSError as exc:
        if exc.errno != os.errno.EEXIST:
          raise

    if ((not os.path.exists(self.model_prefix + '.model')) or
        (not os.path.exists(self.model_prefix + '.vocab')) or
        overwrite):
      sentpiece_train_exec_loc = os.path.join(path, 'spm_train')
      sentpiece_train_command = [sentpiece_train_exec_loc
          , '--input=' + ','.join(train_files)
          , '--model_prefix=' + str(model_prefix)
          , '--vocab_size=' + str(vocab_size)
          , '--model_type=' + str(model_type)
          ]
      subprocess.call(sentpiece_train_command)

    sentpiece_encode_exec_loc = os.path.join(self.sentpiece_path, 'spm_encode')
    sentpiece_encode_command = [sentpiece_encode_exec_loc
        , '--model=' + self.model_prefix + '.model'
        , '--output_format=' + self.output_format
        ] + self.encode_extra_options
    self.tokenizer_command = sentpiece_encode_command

##### Sentence filterers

class SentenceFilterer():
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
    if spec != None:
      for my_spec in spec:
        if my_spec["type"] == "length":
          preproc_list.append(SentenceFiltererLength(my_spec))
        else:
          raise RuntimeError("Unknown preprocessing type {}".format(my_spec["type"]))
    return preproc_list

class SentenceFiltererLength(object):
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
        idx = idx_map.get(idx_map, int(idx))
        if direc == "max":
          self.each_max[idx] = v
        elif direc == "max":
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
    if spec != None:
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
