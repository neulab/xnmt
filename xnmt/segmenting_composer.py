import dynet as dy
import numpy as np

from xnmt.linear import Linear
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref, Path, bare
from xnmt.param_init import GlorotInitializer
from xnmt.events import register_xnmt_handler, register_xnmt_event, handle_xnmt_event

class SegmentComposer(Serializable):
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

class TailSegmentTransformer(Serializable):
  yaml_tag = u"!TailSegmentTransformer"
  @serializable_init
  def __init__(self): pass
  def transform(self, encoder, encodings):
    return encoder.get_final_states()[0].main_expr()

class AverageSegmentTransformer(Serializable):
  yaml_tag = "!AverageSegmentTransformer"
  @serializable_init
  def __init__(self): pass
  def transform(self, encoder, encodings):
    return dy.average(encodings.as_list())

class SumSegmentTransformer(Serializable):
  yaml_tag = "!SumSegmentTransformer"
  @serializable_init
  def __init__(self): pass
  def transform(self, encoder, encodings):
    return dy.sum_dim(encodings.as_tensor(), [1])

class MaxSegmentTransformer(Serializable):
  yaml_tag = "!MaxSegmentTransformer"
  @serializable_init
  def __init__(self): pass
  def transform(self, encoder, encodings):
    return dy.emax(encodings)

class CharNGramSegmentComposer(Serializable):
  yaml_tag = "!CharNGramSegmentComposer"
  
  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               word_vocab=None,
               word_ngram=None,
               ngram_size=4,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               vocab_size=None):
    assert word_vocab is not None or vocab_size is not None, "Can't be both None!"
    if word_vocab is None:
      word_vocab = Vocab()
      dict_entry = vocab_size
    else:
      dict_entry = len(word_vocab)

    self.dict_entry = dict_entry
    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.ngram_size = ngram_size
    self.word_ngram = self.add_serializable_component("word_ngram", word_ngram,
                                                      lambda: Linear(input_dim=dict_entry,
                                                                     output_dim=hidden_dim))
    self.cached_src = None

  @register_xnmt_event
  def set_word_boundary(self, start, end, src):
    self.word = tuple(src[start:end+1])
    if self.cached_src != src:
      self.cached_src = src
      self.src_sent = "".join([self.src_vocab[i] for i in src])
    word_vector = np.zeros(self.dict_entry)
    for i in range(start, end+1):
      for j in range(i, min(i+self.ngram_size, end+1)):
        ngram = self.src_sent[i:j+1]
        if ngram in self.word_vocab:
          word_vector[self.word_vocab.convert(ngram)] += 1
    self.ngram_vocab_vect = dy.inputTensor(word_vector)

  def transduce(self, inputs):
    return dy.tanh(self.word_ngram(self.ngram_vocab_vect))

class WordEmbeddingSegmentComposer(Serializable):
  yaml_tag = "!WordEmbeddingSegmentComposer"

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               word_vocab=None,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               vocab_size=None):
    assert word_vocab is not None or vocab_size is not None, "Can't be both None!"
    param_collection = ParamManager.my_params(self)
    if word_vocab is None:
      word_vocab = Vocab()
      dict_entry = vocab_size
    else:
      dict_entry = len(word_vocab)
    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.embedding = param_collection.add_lookup_parameters((dict_entry, hidden_dim))
    self.cached_src = None

  @register_xnmt_event
  def set_word_boundary(self, start, end, src):
    if self.cached_src != src:
      self.cached_src = src
      self.src_sent = "".join([self.src_vocab[i] for i in src])
    self.word = self.word_vocab.convert(self.src_sent[start:end+1]) # Embedding

  def transduce(self, inputs):
    return self.embedding[self.word]

class ConvolutionSegmentComposer(Serializable):
  yaml_tag = "!ConvolutionSegmentComposer"

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               filter_height,
               filter_width,
               channel,
               num_filter,
               stride,
               dropout_rate = 0,
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               hidden_dim=Ref("exp_global.default_layer_dim")):
    model = ParamManager.my_params(self)
    self.filter = model.add_parameters(dim=(filter_height, filter_width, channel, num_filter),
                                       init=param_init.initializer((filter_height, filter_width, channel, num_filter)))
    self.stride = stride
    self.filter_width = filter_width
    self.filter_height = filter_height
    self.dropout_rate = dropout_rate
    self.train = False

  @register_xnmt_event
  def set_word_boundary(self, start, end, src):
    pass

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
  
  def transduce(self, encodings):
    inp = encodings.as_tensor()
    dim = inp.dim()
    
    if self.train:
      inp = dy.dropout(inp, self.dropout_rate)
    encodings = dy.rectify(dy.conv2d(inp, dy.parameter(self.filter), stride = self.stride, is_valid=False))
    pool = dy.max_dim(encodings, d=1)
    return pool

