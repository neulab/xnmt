import dynet as dy
import pylru

from collections import Counter
from functools import lru_cache

from xnmt.expression_seqs import ExpressionSequence
from xnmt.modelparts.transforms import Linear
from xnmt.param_collections import ParamManager
from xnmt.persistence import serializable_init, Serializable, Ref, Path, bare
from xnmt.param_initializers import GlorotInitializer, ZeroInitializer
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.transducers.recurrent import BiLSTMSeqTransducer

class SingleComposer(object):
  @register_xnmt_handler
  def __init__(self):
    pass

  def compose(self, composed_words, batch_size):
    outputs = [[] for _ in range(batch_size)]
    exprs = []
    # Batching expression
    for expr_list, batch_num, position, start, end in composed_words:
      self.set_word(self.src_sent[batch_num][start:end])
      expr = self.transduce(expr_list)
      if expr is not None:
        outputs[batch_num].append(expr)
        exprs.append(expr)
    dy.forward(exprs)
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
    self.seq_transducer.transduce(ExpressionSequence(expr_tensor=embed))
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

class VocabBasedComposer(SingleComposer):
  @register_xnmt_handler
  def __init__(self,
               vocab,
               vocab_size,
               cache_id_pool,
               cache_word_table):
    self.vocab = vocab
    self.learn_vocab = vocab is None

    cache_size = len(vocab) if vocab is not None else vocab_size
    self.lrucache = pylru.lrucache(cache_size, self.on_word_delete)
    self.cache_id_pool = cache_id_pool or []
    self.cache_word_table = cache_word_table or {}
    self.save_processed_arg("cache_id_pool", self.cache_id_pool)
    self.save_processed_arg("cache_word_table", self.cache_word_table)
    
    # Adding words according to its timestep
    for i, (wordid, (_, word)) in enumerate(sorted(self.cache_word_table.items(), key=lambda x: x[1][0])):
      self.lrucache[word] = int(wordid)
    self.cache_counter = len(self.lrucache)

  def on_word_delete(self, word, wordid):
    if self.learn_vocab:
      self.cache_id_pool.append(wordid)
      self.on_id_delete(wordid)
      del self.cache_word_table[str(wordid)]
    else:
      raise ValueError("Should not delete any id when not learning")
    
  def convert(self, word):
    self.current_word = word
    if self.vocab is not None:
      wordid = self.vocab.convert(word)
    elif word not in self.lrucache:
      if self.train:
        wordid = len(self.lrucache)
        self.lrucache[word] = wordid  # Cache value
        if wordid == self.lrucache.size():
          # Don't merge this line, it will cause bug!
          wordid = self.cache_id_pool.pop()
          self.lrucache[word] =  wordid
      else:
        wordid = self.lrucache.size()  # Unknown ID
    else:
      wordid = self.lrucache[word]
    try:
      return wordid
    finally:
      self.cache_word_table[str(wordid)] = [self.cache_counter, word]
      self.cache_counter += 1
      
  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  def on_id_delete(self, wordid):
    raise NotImplementedError("Should implement process_word")

class LookupComposer(VocabBasedComposer, Serializable):
  yaml_tag = '!LookupComposer'
  @serializable_init
  def __init__(self,
               word_vocab=None,
               vocab_size=32000,
               cache_id_pool=None,
               cache_word_table=None,
               char_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer))):
    super().__init__(word_vocab, vocab_size, cache_id_pool, cache_word_table)
    param_collection = ParamManager.my_params(self)
    # Attributes
    if word_vocab is None:
      self.dict_entry = vocab_size+1
    else:
      self.dict_entry = len(word_vocab)
    self.char_vocab = char_vocab
    self.hidden_dim = hidden_dim
    # Word Embedding
    embed_dim = (self.dict_entry, hidden_dim)
    self.param_init = param_init.initializer(embed_dim, is_lookup=True)
    self.embedding = param_collection.add_lookup_parameters(embed_dim, init=self.param_init)

  def transduce(self, inputs):
    return dy.lookup(self.embedding, self.convert(self.word))

  @lru_cache(maxsize=32000)
  def to_word(self, word):
    return "".join([self.char_vocab[c] for c in word]) 

  def set_word(self, word):
    self.word = self.to_word(word)

  def on_id_delete(self, wordid):
    assert self.train and self.learn_vocab
    # TODO Temporarily reset the value to the values between uniform(-1, 1)
    #new_vct = np.random.uniform(low=-1, high=1, size=self.hidden_dim)
    #self.embedding.init_row(wordid, new_vct)

class CharNGramComposer(VocabBasedComposer, Serializable):
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
               embedding=None,
               ngram_size=4,
               vocab_size=32000,
               cache_id_pool=None,
               cache_word_table=None,
               char_vocab=Ref(Path("model.src_reader.vocab")),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               param_init=Ref("exp_global.param_init", default=bare(GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(ZeroInitializer))):
    super().__init__(word_vocab, vocab_size, cache_id_pool, cache_word_table)
    # Attributes
    if word_vocab is None:
      self.dict_entry = vocab_size+1
    else:
      self.dict_entry = len(word_vocab)
    self.char_vocab = char_vocab
    self.param_init = param_init
    self.bias_init = bias_init
    self.hidden_dim = hidden_dim
    self.word_vect = None
    # Word Embedding
    self.ngram_size = ngram_size
    self.embedding = self.add_serializable_component("embedding", embedding,
                                                      lambda: Linear(input_dim=self.dict_entry,
                                                                     output_dim=hidden_dim,
                                                                     param_init=param_init,
                                                                     bias_init=bias_init))

  def transduce(self, inputs):
    ngrams = [self.convert(ngram) for ngram in self.word_vect.keys()]
    counts = list(self.word_vect.values())
    if len(ngrams) != 0:
      ngram_vocab_vect = dy.sparse_inputTensor([ngrams], counts, (self.dict_entry,))
      return dy.rectify(self.embedding.transform(ngram_vocab_vect))
    else:
      return None

  @lru_cache(maxsize=256000)
  def to_word_vector(self, word):
    word = "".join([self.char_vocab[c] for c in word])
    word_vector = Counter()
    for i in range(len(word)):
      for j in range(i, min(i+self.ngram_size, len(word))):
        ngram = word[i:j+1]
        if self.vocab is None or ngram in self.vocab.w2i:
          word_vector[ngram] += 1
    if self.vocab is not None and len(word_vector) == 0:
      word_vector[self.vocab.UNK_STR] += 1
    return word_vector

  def set_word(self, word):
    self.word_vect = self.to_word_vector(word)

  def on_id_delete(self, wordid):
    assert self.train and self.learn_vocab
    # Temporarily reset the initialization to (-1, 1)
    #W = dy.parameter(self.embedding.W1)
    #b = dy.parameter(self.embedding.b1)

    #W_np = W.as_array()
    #W_np[:,wordid] = np.random.uniform(low=-1, high=1, size=self.hidden_dim)
    #W.set_value(W_np)

    #b_np = b.as_array()
    #b_np[wordid] = 0
    #b.set_value(b_np)

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

