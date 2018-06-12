from xnmt.vocab import Vocab
from xnmt.persistence import Serializable, serializable_init

class Output(object):
  """
  A template class to represent all output.
  """
  def __init__(self, actions=None):
    """ Initialize an output with actions. """
    self.actions = actions or []

  def to_string(self):
    raise NotImplementedError('All outputs must implement to_string.')

class TextOutput(Output):
  def __init__(self, actions=None, vocab=None, score=None):
    self.actions = [] if actions is None else actions
    self.vocab = vocab
    self.score = score
    self.filtered_tokens = {Vocab.SS, Vocab.ES}

  def to_string(self):
    map_func = lambda wi: self.vocab[wi] if self.vocab is not None else str
    return map(map_func, filter(lambda wi: wi not in self.filtered_tokens, self.actions))


class OutputProcessor(object):
  def process_outputs(self, outputs):
    raise NotImplementedError()

  @staticmethod
  def get_output_processor(spec):
    if spec == "none":
      return PlainTextOutputProcessor()
    elif spec == "join-char":
      return JoinCharTextOutputProcessor()
    elif spec == "join-bpe":
      return JoinBPETextOutputProcessor()
    elif spec == "join-piece":
      return JoinPieceTextOutputProcessor()
    elif not isinstance(spec, OutputProcessor):
      raise RuntimeError("Unknown postprocessing argument {}".format(spec))
    else:
      return spec


class PlainTextOutputProcessor(OutputProcessor, Serializable):
  """
  Handles the typical case of writing plain text,
  with one sent per line.
  """
  yaml_tag = "!PlainTextOutputProcessor"
  @serializable_init
  def process_outputs(self, outputs):
    for output in outputs:
      output.plaintext = self.words_to_string(output.to_string())

  def words_to_string(self, word_list):
    return " ".join(word_list)

class JoinCharTextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a single-character vocabulary and joins them to form words;
  per default, double underscores '__' are treated as word separating tokens
  """
  yaml_tag = "!JoinCharTextOutputProcessor"
  @serializable_init
  def __init__(self, space_token="__"):
    self.space_token = space_token

  def words_to_string(self, word_list):
    return "".join(map(lambda s: " " if s==self.space_token else s, word_list))

class JoinBPETextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  """
  yaml_tag = "!JoinBPETextOutputProcessor"
  @serializable_init
  def __init__(self, merge_indicator="@@"):
    self.merge_indicator_with_space = merge_indicator + " "

  def words_to_string(self, word_list):
    return " ".join(word_list).replace(self.merge_indicator_with_space, "")

class JoinPieceTextOutputProcessor(PlainTextOutputProcessor, Serializable):
  """
  Assumes a sentence-piece vocabulary and joins them to form words;
  space_token could be the starting character of a piece
  per default, the u'\u2581' indicates spaces
  """
  yaml_tag = "!JoinPieceTextOutputProcessor"
  @serializable_init
  def __init__(self, space_token="\u2581"):
    self.space_token = space_token

  def words_to_string(self, word_list):
    return "".join(word_list).replace(self.space_token, " ").strip()
