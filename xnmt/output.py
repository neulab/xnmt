from xnmt.vocab import Vocab

class Output(object):
  '''
  A template class to represent all output.
  '''
  def __init__(self, actions=None):
    ''' Initialize an output with actions. '''
    self.actions = actions or []

  def to_string(self):
    raise NotImplementedError('All outputs must implement to_string.')

class TextOutput(Output):
  def __init__(self, actions=None, vocab=None, score=None):
    self.actions = actions or []
    self.vocab = vocab
    self.score = score
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])

  def to_string(self):
    map_func = lambda wi: self.vocab[wi] if self.vocab != None else str
    return map(map_func, filter(lambda wi: wi not in self.filtered_tokens, self.actions))

class OutputProcessor(object):
  def process_outputs(self, outputs):
    raise NotImplementedError()

class PlainTextOutputProcessor(OutputProcessor):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''
  def process_outputs(self, outputs):
    for output in outputs:
      output.plaintext = self.words_to_string(output.to_string())

  def words_to_string(self, word_list):
    return u" ".join(word_list)

class JoinedCharTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a single-character vocabulary and joins them to form words;
  per default, double underscores '__' are treated as word separating tokens
  '''
  def __init__(self, space_token=u"__"):
    self.space_token = space_token

  def words_to_string(self, word_list):
    return u"".join(map(lambda s: u" " if s==self.space_token else s, word_list))

class JoinedBPETextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  '''
  def __init__(self, merge_indicator=u"@@"):
    self.merge_indicator_with_space = merge_indicator + u" "

  def words_to_string(self, word_list):
    return u" ".join(word_list).replace(self.merge_indicator_with_space, u"")

class JoinedPieceTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a sentence-piece vocabulary and joins them to form words;
  space_token could be the starting character of a piece
  per default, the u'\u2581' indicates spaces
  '''
  def __init__(self, space_token=u"\u2581"):
    self.space_token = space_token

  def words_to_string(self, word_list):
    return u"".join(word_list).replace(self.space_token, u" ").strip()


class CcgPieceOutputProcessor(OutputProcessor):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''
  def __init__(self, tag_set, merge_indicator=u"\u2581"):
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])
    self.tag_set = tag_set
    self.merge_indicator = merge_indicator

  def to_string(self):
    words = map(lambda wi: self.vocab[wi], filter(lambda wi: wi not in self.filtered_tokens, self.actions))
    words = [w for w in words if w not in self.tag_set]
    return words

  def process_outputs(self, outputs):
    return [self.words_to_string(output.to_string(self.tag_set)) for output in outputs]

  def words_to_string(self, word_list):
    return u"".join(word_list).replace(self.merge_indicator, u" ").strip()

class TreeHierOutput(Output):
  def __init__(self, actions=None, rule_vocab=None, word_vocab=None):
    self.actions = actions or []
    self.rule_vocab = rule_vocab
    self.word_vocab = word_vocab
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])
  def to_string(self):
    ret = []
    for a in self.actions:
      #if a[0] in self.filtered_tokens: continue
      if a[1]: # if is terminal
        ret.append([self.word_vocab[a[0]], a[2]])
      else:
        ret.append([self.rule_vocab[a[0]], a[2]])
    return ret