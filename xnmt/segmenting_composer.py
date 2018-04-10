import dynet as dy

from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.events import register_xnmt_handler, register_xnmt_event
from xnmt.reports import Reportable

class SegmentComposer(Serializable, Reportable):
  yaml_tag = "!SegmentComposer"

  @register_xnmt_handler
  @serializable_init
  def __init__(self, encoder, transformer):
    self.encoder = encoder
    self.transformer = transformer

  @register_xnmt_event
  def set_word_boundary(self, start, end, src):
    pass

  @property
  def hidden_dim(self):
    return self.encoder.hidden_dim

  def transduce(self, inputs):
    return self.transformer.transform(self.encoder, self.encoder(inputs))

class SegmentTransformer(Serializable):
  def transform(self, encoder, encodings):
    raise RuntimeError("Should call subclass of SegmentTransformer instead")

class TailSegmentTransformer(SegmentTransformer, Serializable):
  yaml_tag = u"!TailSegmentTransformer"
  def transform(self, encoder, encodings):
    return encoder.get_final_states()[0]._main_expr

class TailWordSegmentTransformer(SegmentTransformer):
  yaml_tag = "!TailWordSegmentTransformer"


  def __init__(self, vocab=None, vocab_size=1e6,
               count_file=None, min_count=1, embed_dim=Ref("exp_global.default_layer_dim")):
    assert vocab is not None
    self.vocab = vocab
    self.lookup = ParamManager.my_params(self).add_lookup_parameters((vocab_size, embed_dim))
    self.frequent_words = None

    if count_file is not None:
      print("Reading count reference...")
      frequent_words = set()
      with open(count_file, "r") as fp:
        for line in fp:
          line = line.strip().split("\t")
          cnt = int(line[-1])
          substr = "".join(line[0:-1])
          if cnt >= min_count:
            frequent_words.add(substr)
      self.frequent_words = frequent_words

  def set_word_boundary(self, start, end, src):
    word = tuple(src[start+1:end+1])
    if self.frequent_words is not None and word not in self.frequent_words:
      self.word = self.vocab.convert(self.vocab.UNK_STR)
    else:
      self.word = self.vocab.convert(word)

  def transform(self, encoder, encodings):
    # TODO(philip30): needs to be fixed ?
    return encoder.get_final_states()[0]._main_expr + self.lookup[self.word]

class WordOnlySegmentTransformer(TailWordSegmentTransformer):
  yaml_tag = "!WordOnlySegmentTransformer"
  def transform(self, encoder, encodings, word):
    return self.lookup[self.get_word(word)]

class AverageSegmentTransformer(SegmentTransformer):
  yaml_tag = "!AverageSegmentTransformer"
  def transform(self, encoder, encodings):
    return dy.average(encodings.as_list())

