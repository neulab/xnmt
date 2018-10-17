import functools
import math
import numbers
import subprocess
from typing import List, Optional, Sequence, Union

import numpy as np

from xnmt.vocabs import Vocab
from xnmt.output import OutputProcessor

class Sentence(object):
  """
  A template class to represent a single data example of any type, used for both model input and output.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    score: a score given to this sentence by a model
  """

  def __init__(self, idx: Optional[int] = None, score: Optional[numbers.Real] = None) -> None:
    self.idx = idx
    self.score = score

  def sent_len(self) -> int:
    """
    Return length of input, included padded tokens.

    Returns: length
    """
    raise NotImplementedError("must be implemented by subclasses")

  def len_unpadded(self) -> int:
    """
    Return length of input prior to applying any padding.

    Returns: unpadded length
    """

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'Sentence':
    """
    Return a new, padded version of the sentence (or self if pad_len is zero).

    Args:
      pad_len: number of tokens to append
    Returns:
      padded sentence
    """
    raise NotImplementedError("must be implemented by subclasses")

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'Sentence':
    """
    Create a new, right-truncated version of the sentence (or self if trunc_len is zero).

    Args:
      trunc_len: number of tokens to truncate
    Returns:
      truncated sentence
    """
    raise NotImplementedError("must be implemented by subclasses")

class ReadableSentence(Sentence):
  """
  A base class for sentences based on readable strings.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    score: a score given to this sentence by a model
    output_procs: output processors to be applied when calling sent_str()
  """
  def __init__(self, idx: numbers.Integral, score: Optional[numbers.Real] = None,
               output_procs: Union[OutputProcessor, Sequence[OutputProcessor]] = []) -> None:
    super().__init__(idx=idx, score=score)
    self.output_procs = output_procs

  def str_tokens(self, **kwargs) -> List[str]:
    """
    Return list of readable string tokens.

    Args:
      **kwargs: should accept arbitrary keyword args

    Returns: list of tokens.
    """
    raise NotImplementedError("must be implemented by subclasses")
  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    """
    Return a single string containing the readable version of the sentence.

    Args:
      custom_output_procs: if not None, overwrite the sentence's default output processors
      **kwargs: should accept arbitrary keyword args

    Returns: readable string
    """
    out_str = " ".join(self.str_tokens(**kwargs))
    pps = self.output_procs
    if custom_output_procs is not None:
      pps = custom_output_procs
    if isinstance(pps, OutputProcessor): pps = [pps]
    for pp in pps:
      out_str = pp.process(out_str)
    return out_str
  def __repr__(self):
    return f'"{self.sent_str()}"'
  def __str__(self):
    return self.sent_str()

class ScalarSentence(ReadableSentence):
  """
  A sentence represented by a single integer value, optionally interpreted via a vocab.

  This is useful for classification-style problems.

  Args:
    value: scalar value
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    vocab: optional vocab to give different scalar values a string representation.
    score: a score given to this sentence by a model
  """
  def __init__(self, value: numbers.Integral, idx: Optional[numbers.Integral] = None, vocab: Optional[Vocab] = None,
               score: Optional[numbers.Real] = None) -> None:
    super().__init__(idx=idx, score=score)
    self.value = value
    self.vocab = vocab
  def sent_len(self) -> int:
    return 1
  def len_unpadded(self) -> int:
    return 1
  def create_padded_sent(self, pad_len: numbers.Integral) -> 'ScalarSentence':
    if pad_len != 0:
      raise ValueError("ScalarSentence cannot be padded")
    return self
  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'ScalarSentence':
    if trunc_len != 0:
      raise ValueError("ScalarSentence cannot be truncated")
    return self
  def str_tokens(self, **kwargs) -> List[str]:
    if self.vocab: return [self.vocab[self.value]]
    else: return [str(self.value)]

class CompoundSentence(Sentence):
  """
  A compound sentence contains several sentence objects that present different 'views' on the same data examples.

  Args:
    sents: a list of sentences
  """
  def __init__(self, sents: Sequence[Sentence]) -> None:
    super().__init__(idx=sents[0].idx)
    self.idx = sents[0].idx
    for s in sents[1:]:
      if s.idx != self.idx:
        raise ValueError("CompoundSentence must contain sentences of consistent idx.")
    self.sents = sents
  def sent_len(self) -> int:
    return sum(sent.sent_len() for sent in self.sents)
  def len_unpadded(self) -> int:
    return sum(sent.len_unpadded() for sent in self.sents)
  def create_padded_sent(self, pad_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")
  def create_truncated_sent(self, trunc_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")


class SimpleSentence(ReadableSentence):
  """
  A simple sentence, represented as a list of tokens

  Args:
    words: list of integer word ids
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    vocab: optionally vocab mapping word ids to strings
    score: a score given to this sentence by a model
    output_procs: output processors to be applied when calling sent_str()
    pad_token: special token used for padding
  """
  def __init__(self,
               words: Sequence[numbers.Integral],
               idx: Optional[numbers.Integral] = None,
               vocab: Optional[Vocab] = None,
               score: Optional[numbers.Real] = None,
               output_procs: Union[OutputProcessor, Sequence[OutputProcessor]] = [],
               pad_token: numbers.Integral = Vocab.ES) -> None:
    super().__init__(idx=idx, score=score, output_procs=output_procs)
    self.pad_token = pad_token
    self.words = words
    self.vocab = vocab

  def __getitem__(self, key):
    ret = self.words[key]
    if isinstance(ret, list):  # support for slicing
      return SimpleSentence(words=ret, idx=self.idx, vocab=self.vocab, score=self.score, output_procs=self.output_procs,
                            pad_token=self.pad_token)
    return self.words[key]

  def sent_len(self):
    return len(self.words)

  @functools.lru_cache(maxsize=1)
  def len_unpadded(self):
    return sum(x != self.pad_token for x in self.words)

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'SimpleSentence':
    if pad_len == 0:
      return self
    return self.sent_with_new_words(self.words + [self.pad_token] * pad_len)

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'SimpleSentence':
    if trunc_len == 0:
      return self
    return self.sent_with_words(self.words[:-trunc_len])

  def str_tokens(self, exclude_ss_es=True, exclude_unk=False, exclude_padded=True, **kwargs) -> List[str]:
    exclude_set = set()
    if exclude_ss_es:
      exclude_set.add(Vocab.SS)
      exclude_set.add(Vocab.ES)
    if exclude_unk: exclude_set.add(self.vocab.unk_token)
    # TODO: exclude padded if requested (i.e., all </s> tags except for the first)
    ret_toks =  [w for w in self.words if w not in exclude_set]
    if self.vocab: return [self.vocab[w] for w in ret_toks]
    else: return [str(w) for w in ret_toks]

  def sent_with_new_words(self, new_words):
    return SimpleSentence(words=new_words,
                          idx=self.idx,
                          vocab=self.vocab,
                          score=self.score,
                          output_procs=self.output_procs,
                          pad_token=self.pad_token)

class SegmentedSentence(SimpleSentence):
  def __init__(self, segment=[], **kwargs) -> None:
    super().__init__(**kwargs)
    self.segment = segment

  def sent_with_new_words(self, new_words):
    return SegmentedSentence(words=new_words,
                             idx=self.idx,
                             vocab=self.vocab,
                             score=self.score,
                             output_procs=self.output_procs,
                             pad_token=self.pad_token,
                             segment=self.segment)


class ArraySentence(Sentence):
  """
  A sentence based on a numpy array containing a continuous-space vector for each token.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    nparr: numpy array of dimension num_tokens x token_size
    padded_len: how many padded tokens are contained in the given nparr
    score: a score given to this sentence by a model
  """

  def __init__(self,
               nparr: np.ndarray,
               idx: Optional[numbers.Integral] = None,
               padded_len: int = 0,
               score: Optional[numbers.Real] = None) -> None:
    super().__init__(idx=idx, score=score)
    self.nparr = nparr
    self.padded_len = padded_len

  def __getitem__(self, key):
    assert isinstance(key, numbers.Integral)
    return self.nparr.__getitem__(key)

  def sent_len(self):
    # TODO: check, this seems wrong (maybe need a 'transposed' version?)
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def len_unpadded(self):
    return len(self) - self.padded_len

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'ArraySentence':
    if pad_len == 0:
      return self
    new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:, -1], (self.nparr.shape[0], 1)),
                                                      (self.nparr.shape[0], pad_len)), axis=1)
    return ArraySentence(new_nparr, idx=self.idx, score=self.score, padded_len=self.padded_len + pad_len)

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'ArraySentence':
    if trunc_len == 0:
      return self
    new_nparr = np.asarray(self.nparr[:-trunc_len])
    return ArraySentence(new_nparr, idx=self.idx, score=self.score, padded_len=max(0,self.padded_len - trunc_len))

  def get_array(self):
    return self.nparr

class NbestSentence(SimpleSentence):
  """
  Output in the context of an nbest list.

  Args:
    base_sent: The base sent object
    nbest_id: The sentence id in the nbest list
    print_score: If True, print nbest_id, score, content separated by ``|||``. If False, drop the score.
  """
  def __init__(self, base_sent: SimpleSentence, nbest_id: numbers.Integral, print_score: bool = False) -> None:
    super().__init__(words=base_sent.words, vocab=base_sent.vocab, score=base_sent.score)
    self.base_output = base_sent
    self.nbest_id = nbest_id
    self.print_score = print_score
  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    content_str = super().sent_str(custom_output_procs=custom_output_procs, **kwargs)
    return self._make_nbest_entry(content_str=content_str)
  def _make_nbest_entry(self, content_str: str) -> str:
    entries = [str(self.nbest_id), content_str]
    if self.print_score:
      entries.insert(1, str(self.base_output.score))
    return " ||| ".join(entries)

class LatticeNode(object):
  """
  A lattice node, keeping track of neighboring nodes.

  Args:
    nodes_prev: A list indices of direct predecessors
    nodes_next: A list indices of direct successors
    value: Word id assigned to this node.
    fwd_log_prob: Lattice log probability normalized in forward-direction (successors sum to 1)
    marginal_log_prob: Lattice log probability globally normalized
    bwd_log_prob: Lattice log probability normalized in backward-direction (predecessors sum to 1)
  """
  def __init__(self,
               nodes_prev: Sequence[numbers.Integral],
               nodes_next: Sequence[numbers.Integral],
               value: numbers.Integral,
               fwd_log_prob: Optional[numbers.Real]=None,
               marginal_log_prob: Optional[numbers.Real]=None,
               bwd_log_prob: Optional[numbers.Real]=None) -> None:
    self.nodes_prev = nodes_prev
    self.nodes_next = nodes_next
    self.value = value
    self.fwd_log_prob = fwd_log_prob
    self.marginal_log_prob = marginal_log_prob
    self.bwd_log_prob = bwd_log_prob


class Lattice(ReadableSentence):
  """
  A lattice structure.

  The lattice is represented as a list of nodes, each of which keep track of the indices of predecessor and
  successor nodes.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    nodes: list of lattice nodes
    vocab: vocabulary for word IDs
  """

  def __init__(self, idx: Optional[numbers.Integral], nodes: Sequence[LatticeNode], vocab: Vocab) -> None:
    self.idx = idx
    self.nodes = nodes
    self.vocab = vocab
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

  def __getitem__(self, key: numbers.Integral) -> int:
    """
    Return the value of a particular lattice node.

    Args:
      key: Index of lattice node.

    Returns:
      Value of lattice node with given index.
    """
    node = self.nodes[key]
    if isinstance(node, list):
      # no guarantee that slice is still a consistent graph
      raise ValueError("Slicing not support for lattices.")
    return node.value

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

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'Lattice':
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
                             value=node.value,
                             fwd_log_prob=node.bwd_log_prob,
                             marginal_log_prob=node.marginal_log_prob,
                             bwd_log_prob=node.bwd_log_prob)
      rev_nodes.append(new_node)
    return Lattice(idx=self.idx, nodes=rev_nodes, vocab=self.vocab)

  def str_tokens(self, **kwargs) -> List[str]:
    """
    Return list of readable string tokens.

    Args:
      **kwargs: ignored

    Returns: list of tokens of linearized lattice.
    """
    return [self.vocab.i2w[node.value] for node in self.nodes]

  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    """
    Return a single string containing the readable version of the sentence.

    Args:
      custom_output_procs: ignored
      **kwargs: ignored

    Returns: readable string
    """
    out_str = str([self.str_tokens(**kwargs), [node.nodes_next for node in self.nodes]])
    return out_str

  def plot(self, out_file, show_log_probs=["fwd_log_prob", "marginal_log_prob", "bwd_log_prob"]):
      from graphviz import Digraph
      dot = Digraph(comment='Lattice')
      for i, node in enumerate(self.nodes):
        node_id = i
        log_prob_strings = [f"{math.exp(getattr(node,field)):.3f}" for field in show_log_probs]
        node_label = f"{self.vocab.i2w[node.value]} {'|'.join(log_prob_strings)}"
        node.id = node_id
        dot.node(str(node_id), f"{node_id} : {node_label}")
      for node_i, node in enumerate(self.nodes):
        for node_next in node.nodes_next:
          edge_from, edge_to = node_i, node_next
          dot.edge(str(edge_from), str(edge_to), "")
      try:
          dot.render(out_file)
      except RuntimeError:
          pass
