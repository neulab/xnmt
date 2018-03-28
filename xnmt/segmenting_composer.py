import dynet as dy

import xnmt.linear
import xnmt.embedder

from xnmt.serialize.tree_tools import Ref, Path
from xnmt.serialize.serializable import Serializable
from xnmt.events import register_handler, handle_xnmt_event, register_xnmt_event
from xnmt.reports import Reportable
from xnmt.vocab import Vocab

class SegmentComposer(Serializable, Reportable):
  yaml_tag = "!SegmentComposer"

  def __init__(self, encoder, transformer):
    register_handler(self)
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

  def __init__(self, exp_global=Ref(Path("exp_global")),
                     vocab=Ref(Path("model.src_reader.vocab")),
                     vocab_size=1e6,
                     count_file=None, min_count=1, embed_dim=None):
    assert vocab is not None
    self.vocab = vocab
    embed_dim = embed_dim or xnmt_global.default_layer_dim
    self.lookup = exp_global.dynet_param_collection.param_col.add_lookup_parameters((vocab_size, embed_dim))
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

