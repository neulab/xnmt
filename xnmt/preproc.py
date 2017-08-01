import argparse
import io
import sys
import os.path
import subprocess
import yaml
from serializer import Serializable

##### Preprocessors

class Normalizer(object):
  """A type of normalization to perform to a file. It is initialized first, then expanded."""

  def __init__(self, spec):
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
  
  TODO: only StreamTokenizers are supported by xnmt_preproc.py right now.
  """
  tokenize_by_file = 0
  def tokenize(self, sent):
    raise RuntimeError("Subclasses of Tokenizer must implement tokenize() or tokenize_stream()")

  def tokenize_stream(self, stream):
    """
    Pass a stream to a tokenizer wholesale for efficiency.

    :return: A readable stream providing the tokenized file.

    """
    raise RuntimeError("Subclasses of Tokenizer must implement tokenize() or tokenize_stream()")

  def detokenize(self, sent):
    raise RuntimeError("Subclasses of Tokenizer must implement detokenize() or detokenize_stream()")

class BPETokenizer(Tokenizer):
  """
  Class for byte-pair encoding tokenizer.

  TODO: Unimplemented
  """
  yaml_tag = u'!BPETokenizer'

  def __init__(self, vocab_size, train_files):
    """Determine the BPE based on the vocab size and corpora"""
    pass

  def tokenize(self, sent):
    """Tokenizes a single sentence according to the determined BPE."""
    return ' '.join(['blick' for x in sent.split()])


class StreamTokenizer(Tokenizer):
  """
  Class for tokenizers whose external constraints (e.g. reliance on an external
  executable) demand that they be executed on an entire file/stream at once.
  """

  def tokenize_stream(self, stream):
    """
    Tokenize a file-like text stream.
    
    :param stream: A file-like stream of untokenized text
    :return: A file-like stream of tokenized text

    """
    tokenized_string = self._tokenize(stream.read())
    ram_file = io.StringIO()
    ram_file.write(tokenized_string.decode('utf-8'))
    ram_file.seek(0)
    return ram_file

  def detokenize_stream(self, stream):
    """
    Detokenize a file-like text stream.

    :param stream: A tokenized file-like stream of text
    :return: An untokenized file-like stream of text.

    """
    detokenized_string = self._detokenize(stream.read())
    ram_file = io.StringIO()
    ram_file.write(detokenized_string.decode('utf-8'))
    ram_file.seek(0)
    return ram_file


class ExternalTokenizer(StreamTokenizer):
  """
  Class for arbitrary external tokenizer that accepts untokenized text to stdin and
  emits tokenized tezt to stdout, with passable parameters.

  It is assumed that in general, external tokenizers will be more efficient when run
  once per file, so are run as such (instead of one-execution-per-line.)

  """
  yaml_tag = u'!ExternalTokenizer'
  tokenize_by_file = 1

  def __init__(self, path, tokenizer_args={}, detokenizer_path=None, detokenizer_args={}, arg_separator=' '):
    """Initialize the wrapper around the external tokenizer and optional detokenizer. """
    tokenizer_options = []
    detokenizer_options = []
    if arg_separator != ' ':
      tokenizer_options = [option + arg_separator + str(tokenizer_args[option])
          for option in tokenizer_args]
      detokenizer_options = [option + arg_separator + str(detokenizer_args[option])
          for option in detokenizer_args]
    else:
      for option in tokenizer_args:
        tokenizer_options.extend([option, str(tokenizer_args[option])])
      for option in detokenizer_args:
        detokenizer_options.extend([option, str(tokenizer_args[option])])
    self.tokenizer_command = [path] + tokenizer_options
    self.detokenizer_command = [detokenizer_path] + tokenizer_options if detokenizer_path else []
    print self.tokenizer_command
    print self.detokenizer_command

  def tokenize(self, sent):
    """
    Pass the sentence through the external tokenizer.

    :param sent: An untokenized sentence
    :return: A tokenized sentence

    """
    return self._tokenize(sent)

  def _tokenize(self, string):
    encode_proc = subprocess.Popen(self.tokenizer_command, stdin=subprocess.PIPE
        , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if isinstance(string, unicode):
      string = string.encode('utf-8')
    stdout, stderr = encode_proc.communicate(string)
    if stderr:
      sys.stderr.write(stderr + '\n')
    return stdout

  def detokenize(self, sent):
    """
    Detokenize a sentence or raise an error if no detokenizer was registered.

    :param sent: A single line of tokenized text.
    :return: A single line of detokenized text

    """
    return self._detokenize(sent)

  def _detokenize(self, string):
    """
    Underlying decode call. Raises a RuntimeError
    if no detokenizer was registered with the tokenizer.

    :param string: An arbitrary tokenized string, potentially with newlines.
    :return: A detokenized string, with newlines respected.

    """
    if not self.detokenizer_command:
      raise RuntimeError("No detokenizer registered for this tokenizer.")
    decode_proc = subprocess.Popen(self.detokenize_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (stdout, stderr) = decode_proc.communicate(string)
    if stderr:
      sys.stderr.write(stderr + '\n')
    return stdout


class SentencepieceTokenizer(ExternalTokenizer):
  """
  A wrapper around an independent installation of the sentencepiece tokenizer
  with passable parameters.
  """
  yaml_tag = u'!SentencepieceTokenizer'
  tokenize_by_file = 1

  def __init__(self, path, train_files, vocab_size, overwrite=True, model_name='sentpiece'
      , output_format='piece', model_type='bpe', input_sentence_size=10000000
      , encode_extra_options=None, decode_extra_options=None):
    """
    Initialize the wrapper around sentencepiece and train the tokenizer.

    If overwrite is set to False, learned model will not be overwritten, even if parameters
    are changed.

    "File" output for Sentencepiece written to StringIO temporarily before being written to disk.

    """
    self.sentpiece_path = path
    self.model_path = model_name + '.model'
    self.output_format = output_format
    self.input_format = output_format
    self.encode_extra_options = ['--extra_options='+encode_extra_options] if encode_extra_options else []
    self.decode_extra_options = ['--extra_options='+decode_extra_options] if decode_extra_options else []

    if (not overwrite and os.path.exists(self.model_path)
        and os.path.exists(model_name + '.vocab')):
      return
    sentpiece_train_exec_loc = os.path.join(path, 'spm_train')
    sentpiece_train_command = [sentpiece_train_exec_loc
        , '--input=' + ','.join(train_files)
        , '--model_prefix=' + str(model_name)
        , '--vocab_size=' + str(vocab_size)
        , '--model_type=' + str(model_type)
        ]
    subprocess.call(sentpiece_train_command)

    sentpiece_encode_exec_loc = os.path.join(self.sentpiece_path, 'spm_encode')
    sentpiece_encode_command = [sentpiece_encode_exec_loc
        , '--model=' + self.model_path
        , '--output_format=' + self.output_format
        ] + self.encode_extra_options
    self.tokenizer_command = sentpiece_encode_command

    sentpiece_decode_exec_loc = os.path.join(self.sentpiece_path, 'spm_decode')
    sentpiece_decode_command = [sentpiece_decode_exec_loc
        , '--model=' + self.model_path
        , '--input_format=' + self.input_format
        ] + self.decode_extra_options
    self.detokenizer_command = sentpiece_decode_command


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

    :param sents: A list of parallel sentences.
    :returns: True if they should be used or False if they should be filtered.
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

    :param vocab: A dictionary of vocabulary words with their frequecies.
    :returns: A new dictionary with frequencies containing only the words to leave in the vocabulary.
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
