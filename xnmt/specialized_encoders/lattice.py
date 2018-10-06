import io
import random
from typing import Any, Optional, Sequence
import numbers

import dynet as dy

from xnmt import batchers, events, expression_seqs, input_readers, param_collections, param_initializers, sent, vocabs
from xnmt.modelparts import embedders
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init


class LatticeNode(object):
  """
  A lattice node, keeping track of neighboring nodes.

  Args:
    nodes_prev: A list indices of direct predecessors
    nodes_next: A list indices of direct successors
    value: A value assigned to this node.
  """
  def __init__(self, nodes_prev: Sequence[numbers.Integral], nodes_next: Sequence[numbers.Integral], value: Any) \
          -> None:
    self.nodes_prev = nodes_prev
    self.nodes_next = nodes_next
    self.value = value

  def new_node_with_val(self, value: Any) -> 'LatticeNode':
    """
    Create a new node that has the same location in the lattice but different value.

    Args:
      value: value of new node.

    Returns:
      A new lattice node with given new value and the same predecessors/successors as the current node.
    """
    return LatticeNode(self.nodes_prev, self.nodes_next, value)


class Lattice(sent.Sentence):
  """
  A lattice structure.

  The lattice is represented as a list of nodes, each of which keep track of the indices of predecessor and
  successor nodes.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    nodes: list of lattice nodes
  """

  def __init__(self, idx: Optional[numbers.Integral], nodes: Sequence[LatticeNode]) -> None:
    self.idx = idx
    self.nodes = nodes
    assert len(nodes[0].nodes_prev) == 0
    assert len(nodes[-1].nodes_next) == 0
    for t in range(1, len(nodes) - 1):
      assert len(nodes[t].nodes_prev) > 0
      assert len(nodes[t].nodes_next) > 0
    self.mask = None
    self.expr_tensor = None

  def sent_len(self) -> int:
    """Return number of nodes in the lattice.

    Return:
      Number of nodes in lattice.
    """
    return len(self.nodes)

  def len_unpadded(self) -> int:
    """Return number of nodes in the lattice (padding is not supported with lattices).

    Returns:
      Number of nodes in lattice.
    """
    return self.sent_len()

  def __getitem__(self, key: numbers.Integral) -> LatticeNode:
    """
    Return a particular lattice node.

    Args:
      key: Index of lattice node to return.

    Returns:
      Lattice node with given index.
    """
    ret = self.nodes[key]
    if isinstance(ret, list):
      # no guarantee that slice is still a consistent graph
      raise ValueError("Slicing not support for lattices.")
    return ret

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'Lattice':
    """
    Return self, as padding is not supported.

    Args:
      pad_len: Number of tokens to pad, must be 0.

    Returns:
      self.
    """
    if pad_len != 0: raise ValueError("Lattices cannot be padded.")
    return self

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'Sentence':
    """
    Return self, as truncation is not supported.

    Args:
      trunc_len: Number of tokens to truncate, must be 0.

    Returns:
      self.
    """
    if trunc_len != 0: raise ValueError("Lattices cannot be truncated.")
    return self

  def reversed(self) -> 'Lattice':
    """
    Create a lattice with reversed direction.

    The new lattice will have lattice nodes in reversed order and switched successors/predecessors.

    Returns:
      Reversed lattice.
    """
    rev_nodes = []
    seq_len = len(self.nodes)
    for node in reversed(self.nodes):
      new_node = LatticeNode(nodes_prev=[seq_len - n - 1 for n in node.nodes_next],
                             nodes_next=[seq_len - p - 1 for p in node.nodes_prev],
                             value=node.value)
      rev_nodes.append(new_node)
    return Lattice(idx=self.idx, nodes=rev_nodes)

  def as_list(self) -> list:
    """
    Return list of values.

    Returns:
      List of values.
    """
    return [node.value for node in self.nodes]

  def as_tensor(self) -> dy.Expression:
    """
    Return tensor expression of complete sequence, assuming node values are DyNet vector expressions.

    Returns:
      Lattice as tensor expression.
    """
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate_cols(self.as_list())
    return self.expr_tensor

  def _add_bwd_connections(self, nodes: Sequence[LatticeNode]) -> None:
    """
    Add backward connections, given lattice nodes that specify only forward connections.

    Args:
      nodes: lattice nodes
    """
    for pos in range(len(nodes)):
      for pred_i in nodes[pos].nodes_prev:
        nodes[pred_i].nodes_next.append(pos)
    return nodes

# TODO: remove BinnedLattice
class BinnedLattice(Lattice):
  """
  A binned lattice.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    bins: nested list with indices [bin_pos][rep_pos][token_pos]
  """

  def __init__(self, idx: Optional[int], bins):
    super(BinnedLattice, self).__init__(idx=idx, nodes=self.bins_to_nodes(bins))
    self.bins = bins

  def __repr__(self):
    return str(self.bins)

  def bins_to_nodes(self, bins, drop_arcs=0.0):
    assert len(bins[0]) == len(bins[-1]) == len(bins[0][0]) == len(bins[-1][0]) == 1
    nodes = [LatticeNode([], [], bins[0][0][0])]
    prev_indices = [0]
    for cur_bin in bins[1:-1]:
      new_prev_indices = []
      if drop_arcs > 0.0:
        shuffled_bin = list(cur_bin)
        random.shuffle(shuffled_bin)
        dropped_bin = [shuffled_bin[0]]
        for b in shuffled_bin[1:]:
          if random.random() > drop_arcs:
            dropped_bin.append(b)
        cur_bin = dropped_bin
      for rep in cur_bin:
        for rep_pos in range(len(rep)):
          if rep_pos == 0:
            preds = prev_indices
          else:
            preds = [len(nodes) - 1]
          # print("node", len(nodes), preds)
          nodes.append(LatticeNode(preds, [], rep[rep_pos]))
        new_prev_indices.append(len(nodes) - 1)
      prev_indices = new_prev_indices
    nodes.append(LatticeNode(prev_indices, [], bins[-1][0][0]))
    return self._add_bwd_connections(nodes)

  def drop_arcs(self, dropout):
    return Lattice(nodes=self.bins_to_nodes(self.bins, drop_arcs=dropout))

# TODO: replace by reader that reads actual lattices from file
class LatticeTextReader(input_readers.BaseTextReader, Serializable):
  yaml_tag = '!LatticeTextReader'

  @serializable_init
  def __init__(self, vocab=None, use_words=True, use_chars=False, use_pronun_from=None):
    self.vocab = vocab
    self.use_chars = use_chars
    self.use_words = use_words
    self.use_pronun = False
    if use_pronun_from:
      self.use_pronun = {}
      for l in io.open(use_pronun_from):
        spl = l.strip().split()
        word = spl[0]
        pronun = spl[1:]
        assert word not in self.use_pronun
        self.use_pronun[word] = pronun

  # def read_sents(self, filename, filter_ids=None):
  #   if self.vocab is None:
  #     self.vocab = vocabs.Vocab()
  #   sents = []
  #   for l in self.iterate_filtered(filename, filter_ids):
  #     words = l.strip().split()
  #     if words[0] != vocabs.Vocab.SS_STR: words.insert(0, vocabs.Vocab.SS_STR)
  #     if words[-1] != vocabs.Vocab.ES_STR: words.append(vocabs.Vocab.ES_STR)
  #     bins = []
  #     for word in words:
  #       representations = self.get_representations(word)
  #       cur_bin = []
  #       for rep in representations:
  #         cur_rep_mapped = []
  #         for rep_token in rep:
  #           cur_rep_mapped.append(self.vocab.convert(rep_token))
  #         cur_bin.append(cur_rep_mapped)
  #       bins.append(cur_bin)
  #     lattice = BinnedLattice(bins=bins)
  #     sents.append(lattice)
  #   return sents

  def read_sent(self, line: str, idx: numbers.Integral) -> BinnedLattice:
    words = line.strip().split()
    if words[0] != vocabs.Vocab.SS_STR: words.insert(0, vocabs.Vocab.SS_STR)
    if words[-1] != vocabs.Vocab.ES_STR: words.append(vocabs.Vocab.ES_STR)
    bins = []
    for word in words:
      representations = self.get_representations(word)
      cur_bin = []
      for rep in representations:
        cur_rep_mapped = []
        for rep_token in rep:
          cur_rep_mapped.append(self.vocab.convert(rep_token))
        cur_bin.append(cur_rep_mapped)
      bins.append(cur_bin)
    return BinnedLattice(idx=idx, bins=bins)

  def vocab_size(self):
    return len(self.vocab)

  def get_representations(self, word):
    if word in [vocabs.Vocab.ES_STR, vocabs.Vocab.SS_STR, vocabs.Vocab.UNK_STR]:
      return [[word]]
    reps = []
    if self.use_words:
      reps.append([word])
    if self.use_chars:
      reps.append(["c_" + char for char in word] + ["c_"])
    if self.use_pronun:
      if word in self.use_pronun:
        reps.append(self.use_pronun[word] + ['__'])
      else:
        print("WARNING: no pronunciation for", word)
    return reps


class LatticeEmbedder(embedders.SimpleWordEmbedder, Serializable):
  """
  Simple word embeddings via lookup.

  Args:
    vocab_size:
    emb_dim:
    word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level,
                  see https://arxiv.org/abs/1512.05287
  """

  yaml_tag = '!LatticeEmbedder'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               vocab=None,
               vocab_size=None,
               emb_dim=Ref("exp_global.default_layer_dim"),
               word_dropout=0.0,
               arc_dropout=0.0,
               yaml_path=None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               src_reader=Ref("model.src_reader", default=None),
               trg_reader=Ref("model.trg_reader", default=None)):
    # TODO: refactor by taking a base embedder, and only adding the lattice structure on top of its output?
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.emb_dim = emb_dim
    self.word_dropout = word_dropout
    param_collection = param_collections.ParamManager.my_params(self)
    self.embeddings = param_collection.add_lookup_parameters((self.vocab_size, self.emb_dim),
                                                             init=param_init.initializer(
                                                               (self.vocab_size, self.emb_dim), is_lookup=True))
    self.word_id_mask = None
    self.weight_noise = 0.0
    self.fix_norm = None
    self.arc_dropout = arc_dropout

  def embed_sent(self, sent):
    if batchers.is_batched(sent):
      assert len(sent) == 1, "LatticeEmbedder requires batch size of 1"
      assert sent.mask is None
      sent = sent[0]
    if self.train and self.arc_dropout > 0.0:
      sent = sent.drop_arcs(self.arc_dropout)
    embedded_nodes = [word.new_node_with_val(self.embed(word.value)) for word in sent]
    return Lattice(idx=sent.idx, nodes=embedded_nodes)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val


class LatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A lattice LSTM.

  This is the unidirectional single-layer lattice LSTM.

  Args:
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
  """

  yaml_tag = "!LatticeLSTMTransducer"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0)) -> None:
    self.dropout_rate = dropout
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = param_collections.ParamManager.my_params(self)

    # [i; o; g]
    self.p_Wx_iog = model.add_parameters(dim=(hidden_dim * 3, input_dim))
    self.p_Wh_iog = model.add_parameters(dim=(hidden_dim * 3, hidden_dim))
    self.p_b_iog = model.add_parameters(dim=(hidden_dim * 3,), init=dy.ConstInitializer(0.0))
    self.p_Wx_f = model.add_parameters(dim=(hidden_dim, input_dim))
    self.p_Wh_f = model.add_parameters(dim=(hidden_dim, hidden_dim))
    self.p_b_f = model.add_parameters(dim=(hidden_dim,), init=dy.ConstInitializer(1.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.dropout_mask_x = None
    self.dropout_mask_h = None

  def get_final_states(self):
    return self._final_states

  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def transduce(self, lattice):
    Wx_iog = dy.parameter(self.p_Wx_iog)
    Wh_iog = dy.parameter(self.p_Wh_iog)
    b_iog = dy.parameter(self.p_b_iog)
    Wx_f = dy.parameter(self.p_Wx_f)
    Wh_f = dy.parameter(self.p_Wh_f)
    b_f = dy.parameter(self.p_b_f)
    h = []
    c = []

    batch_size = lattice[0].value.dim()[1]
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    for x_t in lattice:
      val = x_t.value
      if self.dropout_rate > 0.0 and self.train:
        val = dy.cmult(val, self.dropout_mask_x)
      i_ft_list = []
      if len(x_t.nodes_prev) == 0:
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val])
      else:
        h_tilde = sum(h[pred] for pred in x_t.nodes_prev)
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val, Wh_iog, h_tilde])
        for pred in x_t.nodes_prev:
          i_ft_list.append(dy.logistic(dy.affine_transform([b_f, Wx_f, val, Wh_f, h[pred]])))
      i_ait = dy.pick_range(tmp_iog, 0, self.hidden_dim)
      i_aot = dy.pick_range(tmp_iog, self.hidden_dim, self.hidden_dim * 2)
      i_agt = dy.pick_range(tmp_iog, self.hidden_dim * 2, self.hidden_dim * 3)

      i_it = dy.logistic(i_ait)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if len(x_t.nodes_prev) == 0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        fc = dy.cmult(i_ft_list[0], c[x_t.nodes_prev[0]])
        for i in range(1, len(x_t.nodes_prev)):
          fc += dy.cmult(i_ft_list[i], c[x_t.nodes_prev[i]])
        c.append(fc + dy.cmult(i_it, i_gt))
      h_t = dy.cmult(i_ot, dy.tanh(c[-1]))
      if self.dropout_rate > 0.0 and self.train:
        h_t = dy.cmult(h_t, self.dropout_mask_h)
      h.append(h_t)
    self._final_states = [transducers.FinalTransducerState(h[-1], c[-1])]
    return Lattice(idx=lattice.idx, nodes=[node_t.new_node_with_val(h_t) for node_t, h_t in zip(lattice.nodes, h)])


class BiLatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A multi-layered bidirectional lattice LSTM.

  Makes use of several LatticeLSTMTransducer instances and combines them appropriately.

  Args:
    layers: number of layers
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
    forward_layers: determined automatically
    backward_layers: determined automatically
  """

  yaml_tag = '!BiLatticeLSTMTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               forward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None,
               backward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None) -> None:
    self.num_layers = layers
    input_dim = input_dim
    hidden_dim = hidden_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers",
                                                          forward_layers,
                                                          lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                        hidden_dim=hidden_dim,
                                                                                        dropout=dropout,
                                                                                        layers=layers))
    self.backward_layers = self.add_serializable_component("backward_layers",
                                                           backward_layers,
                                                           lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                         hidden_dim=hidden_dim,
                                                                                         dropout=dropout,
                                                                                         layers=layers))

  def _make_dir_layers(self, input_dim, hidden_dim, dropout, layers):
    dir_layers = [LatticeLSTMTransducer(input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout)]
    dir_layers += [LatticeLSTMTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout) for
                            _ in range(layers - 1)]
    return dir_layers

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def transduce(self, lattice):
    if isinstance(lattice, expression_seqs.ExpressionSequence):
      lattice = Lattice(
        idx=lattice.idx,
        nodes=[LatticeNode([i - 1] if i > 0 else [], [i + 1] if i < len(lattice) - 1 else [], value) \
               for (i, value) in enumerate(lattice)])

    # first layer
    forward_es = self.forward_layers[0].transduce(lattice)
    rev_backward_es = self.backward_layers[0].transduce(lattice.reversed())

    for layer_i in range(1, len(self.forward_layers)):
      concat_fwd = Lattice(
        idx=lattice.idx,
        nodes=[node_fwd.new_node_with_val(dy.concatenate([node_fwd.value, node_bwd.value])) \
               for node_fwd, node_bwd in zip(forward_es, reversed(rev_backward_es.nodes))])
      concat_bwd = Lattice(
        idx=lattice.idx,
        nodes=[node_bwd.new_node_with_val(dy.concatenate([node_fwd.value, node_bwd.value])) \
               for node_fwd, node_bwd in zip(reversed(forward_es.nodes), rev_backward_es)])
      new_forward_es = self.forward_layers[layer_i].transduce(concat_fwd)
      rev_backward_es = self.backward_layers[layer_i].transduce(concat_bwd)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return Lattice(
      idx=lattice.idx,
      nodes=[lattice.nodes[i].new_node_with_val(dy.concatenate([forward_es[i].value, rev_backward_es[-i - 1].value]))
             for i in range(forward_es.sent_len())])
