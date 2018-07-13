import dynet as dy
import numpy as np
from collections import Counter

from xnmt.transform import Linear
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref, Path, bare
from xnmt.param_init import GlorotInitializer
from xnmt.events import register_xnmt_handler, register_xnmt_event, handle_xnmt_event
from xnmt.lstm import BiLSTMSeqTransducer

class SegmentComposer(Serializable):
  yaml_tag = "!SegmentComposer"

  @serializable_init
  def __init__(self, encoder=bare(BiLSTMSeqTransducer),
                     transformer=bare(TailSegmentTransformer)):
    self.encoder = encoder
    self.transformer = transformer

  def set_word_boundary(self, start, end, src):
    pass

  @property
  def hidden_dim(self):
    return self.encoder.hidden_dim

  def transduce(self, inputs):
    return self.transformer.transform(self.encoder, self.encoder.transduce(inputs))

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
    return dy.emax(encodings.as_list())

class CharNGramSegmentComposer(Serializable):
  yaml_tag = "!CharNGramSegmentComposer"
  
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
      word_vocab.freeze()
      if word_vocab.UNK_STR not in word_vocab.w2i:
        dict_entry += 1
      word_vocab.set_unk(word_vocab.UNK_STR)

    self.dict_entry = dict_entry
    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.ngram_size = ngram_size
    self.word_ngram = self.add_serializable_component("word_ngram", word_ngram,
                                                      lambda: Linear(input_dim=dict_entry,
                                                                     output_dim=hidden_dim))
    self.cached_src = None

  def set_word_boundary(self, start, end, src):
    self.word = tuple(src[start:end+1])
    if self.cached_src != src:
      self.cached_src = src
      self.src_sent = "".join([self.src_vocab[i] for i in src])
    word_vector = Counter()
    for i in range(start, end+1):
      for j in range(i, min(i+self.ngram_size, end+1)):
        ngram = self.src_sent[i:j+1]
        if ngram in self.word_vocab:
          word_vector[int(self.word_vocab.convert(ngram))] += 1
    keys = [x for x in word_vector.keys()]
    values = [x for x in word_vector.values()]
    if len(keys) == 0:
      keys = [self.word_vocab.unk_token]
      values = [1]

    self.ngram_vocab_vect = dy.sparse_inputTensor([keys], values, (self.dict_entry,))

  def transduce(self, inputs):
    return dy.rectify(self.word_ngram(self.ngram_vocab_vect))

class WordEmbeddingSegmentComposer(Serializable):
  yaml_tag = "!WordEmbeddingSegmentComposer"

  @serializable_init
  def __init__(self,
               word_vocab=None,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               vocab_size=25000):
    param_collection = ParamManager.my_params(self)
    if word_vocab is None:
      word_vocab = Vocab()
      dict_entry = vocab_size
    else:
      dict_entry = len(word_vocab)
      word_vocab.freeze()
      if word_vocab.UNK_STR not in word_vocab.w2i:
        dict_entry += 1
      word_vocab.set_unk(word_vocab.UNK_STR)

    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.embedding = param_collection.add_lookup_parameters((dict_entry, hidden_dim))
    self.cached_src = None

  def set_word_boundary(self, start, end, src):
    if self.cached_src != src:
      self.cached_src = src
      self.src_sent = tuple([self.src_vocab[i] for i in src])

    self.word = " ".join(self.src_sent[start:end+1])
    self.word_id = self.word_vocab.convert(self.word) # Embedding

  def transduce(self, inputs):
    return self.embedding[self.word_id]

class ConvolutionSegmentComposer(Serializable):
  yaml_tag = "!ConvolutionSegmentComposer"

  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               ngram_size,
               dropout = Ref("exp_global.dropout", default=0.0),
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               embed_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim")):
    model = ParamManager.my_params(self)
    dim = (1, ngram_size, embed_dim, hidden_dim)
    self.filter = model.add_parameters(dim=dim, init=param_init.initializer(dim))
    self.dropout = dropout
    self.train = False
    self.ngram_size = ngram_size
    self.embed_dim = embed_dim

  def set_word_boundary(self, start, end, src):
    pass

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
  
  def transduce(self, encodings):
    inp = encodings.as_tensor()
    dim = inp.dim()
    
    if self.train:
      inp = dy.dropout(inp, self.dropout)
    
    # Padding
    if dim[0][1] < self.ngram_size:
      pad = dy.zeros((self.embed_dim, self.ngram_size-dim[0][1]))
      inp = dy.concatenate([inp, pad], d=1)
      dim = inp.dim()
    inp = dy.reshape(dy.transpose(inp), (1, dim[0][1], dim[0][0]))
    encodings = dy.rectify(dy.conv2d(inp, dy.parameter(self.filter), stride=(1,1), is_valid=True))
    return dy.transpose(dy.max_dim(encodings, d=1))

class SumMultipleSegmentComposer(Serializable):
  yaml_tag = "!SumMultipleSegmentComposer"
  
  @serializable_init
  def __init__(self, segment_composers):
    self.segment_composers = segment_composers

  def set_word_boundary(self, start, end, src):
    for segment_composer in self.segment_composers:
      segment_composer.set_word_boundary(start, end, src)

  def transduce(self, inputs):
    return sum([segment_composer.transduce(inputs) for segment_composer in self.segment_composers])

