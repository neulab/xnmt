import dynet as dy
import numpy as np
from collections import Counter
from functools import lru_cache

from xnmt.expression_seqs import ExpressionSequence
from xnmt.transforms import Linear
from xnmt.param_collections import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref, Path, bare
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.events import register_xnmt_handler, register_xnmt_event, handle_xnmt_event
from xnmt.recurrent import BiLSTMSeqTransducer

class SingleComposer(object):
  @register_xnmt_handler
  def __init__(self):
    pass

  def compose(self, composed_words, sample_size, batch_size):
    outputs = [[[] for j in range(batch_size)] for i in range(sample_size)]
    # Batching expression
    for expr_list, sample_num, batch_num, position, start, end in composed_words:
      self.set_word(self.src_sent[batch_num][start:end])
      outputs[sample_num][batch_num].append(self.transduce(expr_list))
    return outputs

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src_sent = src

  def set_word(self, word):
    pass

  def transduce(self, embeds):
    raise NotImplementedError()

class SumComposer(SingleComposer, Serializable):
  yaml_tag = "!SumComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.sum_dim(embeds, [1])

class AverageComposer(SingleComposer, Serializable):
  yaml_tag = "!AverageComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.mean_dim(embeds, [1], False)

class MaxComposer(SingleComposer, Serializable):
  yaml_tag = "!MaxComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.max_dim(embeds, d=1)

class SeqTransducerComposer(SingleComposer, Serializable):
  yaml_tag = "!SeqTransducerComposer"
  @serializable_init
  def __init__(self, seq_transducer=bare(BiLSTMSeqTransducer)):
    super().__init__()
    self.seq_transducer = seq_transducer

  def transduce(self, embed):
    encodings = self.seq_transducer.transduce(ExpressionSequence(expr_tensor=embed))
    return self.seq_transducer.get_final_states()[-1].main_expr()

class ConvolutionComposer(SingleComposer, Serializable):
  yaml_tag = "!ConvolutionComposer"
  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               ngram_size,
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.param_init", default=bare(ZeroInitializer)),
               embed_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim")):
    model = ParamManager.my_params(self)
    dim = (1, ngram_size, embed_dim, hidden_dim)
    self.filter = model.add_parameters(dim=dim, init=param_init.initializer(dim))
    self.bias = model.add_parameters(dim=(embed_dim,), init=bias_init.initializer(dim))
    self.ngram_size = ngram_size
    self.embed_dim = embed_dim

  def transduce(self, encodings):
    inp = encodings
    dim = inp.dim()
    if dim[0][1] < self.ngram_size:
      pad = dy.zeros((self.embed_dim, self.ngram_size-dim[0][1]))
      inp = dy.concatenate([inp, pad], d=1)
      dim = inp.dim()
    inp = dy.reshape(inp, (1, dim[0][1], dim[0][0]))
    encodings = dy.rectify(dy.conv2d_bias(inp, dy.parameter(self.filter), dy.parameter(self.bias), stride=(1, 1), is_valid=True))
    return dy.max_dim(dy.max_dim(encodings, d=1), d=0)

class LookupComposer(SingleComposer, Serializable):
  yaml_tag = '!LookupComposer'
  @serializable_init
  def __init__(self,
               word_vocab=None,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               vocab_size=25000):
    super().__init__()
    param_collection = ParamManager.my_params(self)
    if word_vocab is None:
      word_vocab = Vocab()
      dict_entry = vocab_size
    else:
      word_vocab.freeze()
      word_vocab.set_unk(word_vocab.UNK_STR)
      dict_entry = len(word_vocab)
    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.embedding = param_collection.add_lookup_parameters((dict_entry, hidden_dim))

  def transduce(self, inputs):
    return dy.lookup(self.embedding, self.word_id)

  def set_word(self, word):
    self.word_id = self.to_word_id(word)

  @lru_cache(maxsize=32000)
  def to_word_id(self, word):
    return self.word_vocab.convert("".join([self.src_vocab[c] for c in word]))

class CharNGramComposer(SingleComposer, Serializable):
  """
  CHARAGRAM composition function

  Args:
    word_vocab: Count of ngrams as vocabulary. Made by running script/vocab/count-charngram.py on the source corpus.
    ngram_size: The limit of ngram window.
  """
  yaml_tag = "!CharNGramComposer"

  @serializable_init
  def __init__(self,
               word_vocab=None,
               ngram_size=4,
               src_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               word_ngram=None,
               vocab_size=None):
    super().__init__()
    if word_vocab is None:
      word_vocab = Vocab()
      dict_entry = vocab_size
    else:
      word_vocab.freeze()
      word_vocab.set_unk(word_vocab.UNK_STR)
      dict_entry = len(word_vocab)

    self.dict_entry = dict_entry
    self.src_vocab = src_vocab
    self.word_vocab = word_vocab
    self.ngram_size = ngram_size
    self.word_ngram = self.add_serializable_component("word_ngram", word_ngram,
                                                      lambda: Linear(input_dim=dict_entry,
                                                                     output_dim=hidden_dim))

  @lru_cache(maxsize=32000)
  def to_word_vector(self, word):
    word = "".join([self.src_vocab[c] for c in word])
    word_vector = Counter()
    for i in range(len(word)):
      for j in range(i, min(i+self.ngram_size, len(word))):
        ngram = word[i:j+1]
        if ngram in self.word_vocab.w2i:
          word_vector[int(self.word_vocab.convert(ngram))] += 1
    if len(word_vector) == 0:
      word_vector[int(self.word_vocab[self.word_vocab.UNK_STR])] += 1
    return word_vector

  def transduce(self, inputs):
    keys = list(self.word_vect.keys())
    values = list(self.word_vect.values())
    ngram_vocab_vect = dy.sparse_inputTensor([keys], values, (self.dict_entry,))
    return dy.rectify(self.word_ngram(ngram_vocab_vect))

  def set_word(self, word):
    self.word_vect = self.to_word_vector(word)

class SumMultipleComposer(SingleComposer, Serializable):
  yaml_tag = "!SumMultipleComposer"

  @serializable_init
  def __init__(self, composers):
    super().__init__()
    self.composers = composers

  def set_word(self, word):
    for composer in self.composers:
      composer.set_word(word)

  def transduce(self, embeds):
    return sum([composer.transduce(embeds) for composer in self.composers])

