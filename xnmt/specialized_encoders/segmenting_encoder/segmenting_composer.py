import dynet as dy
import numpy as np
from collections import Counter
from functools import lru_cache

from xnmt.expression_sequence import ExpressionSequence
from xnmt.transform import Linear
from xnmt.param_collection import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref, Path, bare
from xnmt.param_init import GlorotInitializer, ZeroInitializer
from xnmt.events import register_xnmt_handler, register_xnmt_event, handle_xnmt_event
from xnmt.lstm import BiLSTMSeqTransducer

class Composer(object):
  @register_xnmt_handler
  def __init__(self):
    pass

  def compose(self, composed_words, sample_size, batch_size):
    batches = []
    batch_maps = []
    batch_words = []
    seq_len = np.zeros((sample_size, batch_size), dtype=int)
    composed_words = sorted(composed_words, key=lambda x: x[5]-x[4])
    # Batching expression
    now_length = -1
    for expr_list, sample_num, batch_num, position, start, end in composed_words:
      length = end-start
      if length != now_length:
        now_length = length
        now_map = {}
        now_batch = []
        now_words = []
        now_idx = 0
        batches.append(now_batch)
        batch_maps.append(now_map)
        batch_words.append(now_words)
      now_batch.append(expr_list)
      now_words.append(self.src_sent[batch_num][start:end])
      now_map[now_idx] = (sample_num, batch_num, position)
      seq_len[sample_num,batch_num] += 1
      now_idx += 1
    # Composing
    outputs = [[[None for _ in range(seq_len[i,j])] for j in range(batch_size)] for i in range(sample_size)]
    expr_list = []
    for batch, batch_map, batch_word in zip(batches, batch_maps, batch_words):
      self.set_words(batch_word)
      results = self.transduce(dy.concatenate_to_batch(batch))
      results.value()
      for idx, (sample_num, batch_num, position) in batch_map.items():
        expr_list.append(dy.pick_batch_elem(results, idx))
        outputs[sample_num][batch_num][position] = expr_list[-1]
    dy.forward(expr_list)
    return outputs

  @handle_xnmt_event
  def on_start_sent(self, src):
    self.src_sent = src

  def set_words(self, words):
    pass

  def transduce(self, embeds):
    raise NotImplementedError()

class SumComposer(Composer, Serializable):
  yaml_tag = "!SumComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.sum_dim(embeds, [1])

class AverageComposer(Composer, Serializable):
  yaml_tag = "!AverageComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.mean_dim(embeds, [1], False)

class MaxComposer(Composer, Serializable):
  yaml_tag = "!MaxComposer"
  @serializable_init
  def __init__(self):
    super().__init__()

  def transduce(self, embeds):
    return dy.max_dim(embeds, 1)

class SeqTransducerComposer(Composer, Serializable):
  yaml_tag = "!SeqTransducerComposer"
  @serializable_init
  def __init__(self, seq_transducer=bare(BiLSTMSeqTransducer)):
    super().__init__()
    self.seq_transducer = seq_transducer

  def transduce(self, embeds):
    expr_seq = []
    seq_len = embeds.dim()[0][1]
    for i in range(seq_len):
      expr_seq.append(dy.max_dim(dy.select_cols(embeds, [i]), 1))
    encodings = self.seq_transducer.transduce(ExpressionSequence(expr_seq))
    return self.seq_transducer.get_final_states()[-1].main_expr()

class ConvolutionComposer(Composer, Serializable):
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
    inp = dy.reshape(inp, (1, dim[0][1], dim[0][0]), batch_size=dim[1])
    encodings = dy.rectify(dy.conv2d_bias(inp, dy.parameter(self.filter), dy.parameter(self.bias), stride=(1, 1), is_valid=True))
    return dy.max_dim(dy.max_dim(encodings, d=1), d=0)

class LookupComposer(Composer, Serializable):
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
    word_ids = [self.word_vocab.convert(w) for w in self.words]
    return dy.lookup_batch(self.embedding, word_ids)

  def set_words(self, words):
    self.words = ["".join([self.src_vocab[c] for c in word_ids]) for word_ids in words]

class CharNGramComposer(Composer, Serializable):
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
    batch_size = len(self.words)
    word_vects = []
    keys = []
    values = []
    for i, word in enumerate(self.words):
      word_vects.append(self.to_word_vector(word))
    idxs = [(x, i) for i in range(batch_size) for x in word_vects[i].keys()]
    idxs = tuple(map(list, list(zip(*idxs))))

    values = [x for i in range(batch_size) for x in word_vects[i].values()]
    ngram_vocab_vect = dy.sparse_inputTensor(idxs, values, (self.dict_entry, len(self.words )), batched=True)

    return dy.rectify(self.word_ngram(ngram_vocab_vect))

  def set_words(self, words):
    self.words = ["".join([self.src_vocab[c] for c in word_ids]) for word_ids in words]

class SumMultipleComposer(Composer, Serializable):
  yaml_tag = "!SumMultipleComposer"

  @serializable_init
  def __init__(self, composers):
    super().__init__()
    self.composers = composers

  def set_words(self, words):
    for composer in self.composers:
      composer.set_words(words)

  def transduce(self, inputs):
    return sum([composer.transduce(inputs) for composer in self.composers])

