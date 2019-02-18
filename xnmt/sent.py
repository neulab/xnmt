import copy
import functools
import numbers
import enum
from typing import List, Optional, Sequence, Union

import numpy as np

from xnmt.vocabs import Vocab
from xnmt.output import OutputProcessor
from xnmt.graph import  HyperGraph, HyperNode, HyperEdge

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

  def __getitem__(self, key):
    """
    Get an item or a slice of the sentence.

    Args:
      key: index or slice

    Returns:
      A single word or a Sentence object, depending on whether an index or a slice was given as key.
    """
    raise NotImplementedError("must be implemented by subclasses")

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

  def get_unpadded_sent(self) -> 'Sentence':
    """
    Return the unpadded sentence.

    If self is unpadded, return self, if not return reference to original unpadded sentence if possible, otherwise
    create a new sentence.
    """
    if self.sent_len() == self.len_unpadded():
      return self
    else:
      return self[:self.len_unpadded()]

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
  def __getitem__(self, key):
    if isinstance(key, numbers.Integral):
      if key!=0: raise IndexError()
      return self.value
    else:
      if not isinstance(key, slice):
        raise TypeError()
      if key.start!=0 and key.stop!=1: raise IndexError()
      return self
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
  def get_unpadded_sent(self):
    return self # scalar sentences are always unpadded
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
  def __getitem__(self, item):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")
  def sent_len(self) -> int:
    return sum(sent.sent_len() for sent in self.sents)
  def len_unpadded(self) -> int:
    return sum(sent.len_unpadded() for sent in self.sents)
  def create_padded_sent(self, pad_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")
  def create_truncated_sent(self, trunc_len):
    raise ValueError("not supported with CompoundSentence, must be called on one of the sub-inputs instead.")
  def get_unpadded_sent(self):
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
    unpadded_sent: reference to original, unpadded sentence if available
  """
  def __init__(self,
               words: Sequence[numbers.Integral],
               idx: Optional[numbers.Integral] = None,
               vocab: Optional[Vocab] = None,
               score: Optional[numbers.Real] = None,
               output_procs: Union[OutputProcessor, Sequence[OutputProcessor]] = [],
               pad_token: numbers.Integral = Vocab.ES,
               unpadded_sent: 'SimpleSentence' = None) -> None:
    super().__init__(idx=idx, score=score, output_procs=output_procs)
    self.pad_token = pad_token
    self.words = words
    self.vocab = vocab
    self.unpadded_sent = unpadded_sent

  def __getitem__(self, key):
    ret = self.words[key]
    if isinstance(ret, list):  # support for slicing
      return SimpleSentence(words=ret, idx=self.idx, vocab=self.vocab, score=self.score, output_procs=self.output_procs,
                            pad_token=self.pad_token, unpadded_sent=self.unpadded_sent)
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

  def get_unpadded_sent(self):
    if self.unpadded_sent: return self.unpadded_sent
    else: return super().get_unpadded_sent()

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
    unpadded_sent = self.unpadded_sent
    if not unpadded_sent:
      if self.sent_len()==self.len_unpadded(): unpadded_sent = self
    return SimpleSentence(words=new_words,
                          idx=self.idx,
                          vocab=self.vocab,
                          score=self.score,
                          output_procs=self.output_procs,
                          pad_token=self.pad_token,
                          unpadded_sent=unpadded_sent)

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
                             segment=self.segment,
                             unpadded_sent=self.unpadded_sent)

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
               padded_len: numbers.Integral= 0,
               score: Optional[numbers.Real] = None,
               unpadded_sent: 'ArraySentence' = None) -> None:
    super().__init__(idx=idx, score=score)
    self.nparr = nparr
    self.padded_len = padded_len
    self.unpadded_sent = unpadded_sent

  def __getitem__(self, key):
    if not isinstance(key, numbers.Integral): raise NotImplementedError()
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
    return ArraySentence(new_nparr, idx=self.idx, score=self.score, padded_len=self.padded_len + pad_len,
                         unpadded_sent=self if self.padded_len==0 else self.unpadded_sent)

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'ArraySentence':
    if trunc_len == 0:
      return self
    new_nparr = np.asarray(self.nparr[:-trunc_len])
    return ArraySentence(new_nparr, idx=self.idx, score=self.score, padded_len=max(0,self.padded_len - trunc_len),
                         unpadded_sent=self if self.padded_len == 0 else self.unpadded_sent)

  def get_unpadded_sent(self):
    if self.padded_len==0: return self
    elif self.unpadded_sent: return self.unpadded_sent
    else: return super().get_unpadded_sent()

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


class GraphSentence(ReadableSentence):
  """
  A graph structure.

  This is a wrapper for a graph datastructure.

  Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    graph: hypergraph containing graphs
    vocab: vocabulary for word IDs
    num_padded: denoting that this many words are padded (without adding any physical nodes)
    unpadded_sent: reference to original, unpadded sentence if available
  """

  def __init__(self,
               idx: Optional[numbers.Integral],
               score: Optional[numbers.Real],
               graph: HyperGraph,
               value_vocab: Vocab,
               node_vocab: Optional[Vocab] = None,
               edge_vocab: Optional[Vocab] = None,
               num_padded: numbers.Integral = 0,
               unpadded_sent: 'GraphSentence' = None) -> None:
    super().__init__(idx=idx, score=score)
    self.idx = idx
    self.graph = graph
    self.edge_vocab = edge_vocab
    self.node_vocab = node_vocab
    self.value_vocab = value_vocab
    self.num_padded = num_padded
    self.unpadded_sent = unpadded_sent
    
  @functools.lru_cache(maxsize=1)
  def topo_sort(self):
    """
    Return list of nodes in topological sorting ordering.
    """
    return self.graph.topo_sort()

  def sent_len(self) -> int:
    """Return number of nodes in the graph, including padded words.

    Return:
      Number of nodes in graph.
    """
    return self.graph.len_nodes + self.num_padded

  def len_unpadded(self) -> int:
    """Return number of nodes in the graph, without counting padded words.

    Returns:
      Number of nodes in graph.
    """
    return self.graph.len_nodes

  def __getitem__(self, key: numbers.Integral) -> Optional[int]:
    """
    Return the value of a particular graph node. Padded nodes are virtually appended at the end.

    Args:
      key: Index of graph node.

    Returns:
      Value of graph node with given index, or ES if accessing a padded lattice node.
    """
    if self.len_unpadded() <= key < self.sent_len():
      return self.value_vocab.ES
    node = self.graph[key]
    if isinstance(node, list):
      # no guarantee that slice is still a consistent graph
      raise ValueError("Slicing not support for graphs.")
    return node.value

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'GraphSentence':
    """
    Return padded graph.

    Args:
      pad_len: Number of tokens to pad.

    Returns:
      New padded graph, or self if pad_len==0.
    """
    if pad_len == 0:
      return self
    copied_graph = copy.deepcopy(self.graph)
    return GraphSentence(idx=self.idx,
                         score=self.score,
                         graph=copied_graph,
                         value_vocab=self.value_vocab,
                         edge_vocab=self.edge_vocab,
                         node_vocab=self.node_vocab,
                         num_padded=pad_len,
                         unpadded_sent=self.unpadded_sent or super().get_unpadded_sent())

  def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'GraphSentence':
    """
    Return self, as truncation is not supported.

    Args:
      trunc_len: Number of tokens to truncate, must be 0.

    Returns:
      self.
    """
    if trunc_len != 0: raise ValueError("Lattices cannot be truncated.")
    return self

  def get_unpadded_sent(self) -> 'GraphSentence':
    return self.unpadded_sent or super().get_unpadded_sent()

  def reversed(self) -> 'GraphSentence':
    """
    Create a graph with reversed direction.

    The new graph will have graph nodes in reversed order and switched successors/predecessors.
    It will have the same number of padded nodes (again at the end of the nodes!).

    Returns:
      Reversed graph.
    """
    return GraphSentence(idx=self.idx,
                         score=self.score,
                         graph=self.graph.reverse(),
                         value_vocab=self.value_vocab,
                         node_vocab=self.node_vocab,
                         edge_vocab=self.edge_vocab,
                         num_padded=self.num_padded)

  def str_tokens(self, **kwargs) -> List[str]:
    """
    Return list of readable string tokens.

    Args:
      **kwargs: ignored

    Returns: list of tokens of linearized graph.
    """
    return [self.value_vocab.i2w[self.graph[node_id].value] for node_id in self.nodes]

  def sent_str(self, custom_output_procs=None, **kwargs) -> str:
    """
    Return a single string containing the readable version of the sentence.

    Args:
      custom_output_procs: ignored
      **kwargs: ignored

    Returns: readable string
    """
    out_str = str([self.str_tokens(**kwargs), [self.graph.sucessors(node_id) for node_id in self.nodes]])
    return out_str
  
  def plot(self, out_file):
    try:
      from graphviz import Digraph
    except:
      raise RuntimeError("Need graphviz package to be installed.")
    dot = Digraph(comment='Graph')
    for node_id in self.nodes:
      node = self.graph[node_id]
      node_label = "{} {}".format(self.value_vocab.i2w[node.value], node.feature_str())
      dot.node(str(node_id), "{} : {}".format(node_id, node_label))
    for edge in self.graph.iter_edges():
      for node_next in edge.node_to:
        dot.edge(str(edge.node_from.node_id), str(node_next.node_id), "")
    try:
      dot.render(out_file)
    except RuntimeError:
      pass


class LatticeNode(HyperNode):
  """
  A lattice node.

  Args:
    node_id: Unique identifier for node
    value: Word id assigned to this node.
    fwd_log_prob: Lattice log probability normalized in forward-direction (successors sum to 1)
    marginal_log_prob: Lattice log probability globally normalized
    bwd_log_prob: Lattice log probability normalized in backward-direction (predecessors sum to 1)
  """
  def __init__(self,
               node_id: int,
               value: numbers.Integral,
               fwd_log_prob: Optional[numbers.Real]=0,
               marginal_log_prob: Optional[numbers.Real]=0,
               bwd_log_prob: Optional[numbers.Real]=0) -> None:
    super().__init__(value, node_id)
    self.fwd_log_prob = fwd_log_prob
    self.marginal_log_prob = marginal_log_prob
    self.bwd_log_prob = bwd_log_prob

  def reset_prob(self):
    self.fwd_log_prob = 0
    self.marginal_log_prob = 0
    self.bwd_log_prob = 0

  def reversed(self):
    return LatticeNode(self.node_id, self.value, self.bwd_log_prob, self.marginal_log_prob, self.fwd_log_prob)

  def feature_str(self):
    return "{:.3f}|{:.3f}|{:.3f}".format(self.fwd_log_prob, self.marginal_log_prob, self.bwd_log_prob)

    
class SyntaxTreeNode(HyperNode):
  class Type(enum.Enum):
    NONE=0
    NT=1
    PRT=2
    T=3
  
  def __init__(self, node_id, value, head, node_type=Type.NONE):
    super().__init__(value, node_id)
    self._head = head
    self._type = node_type
    
  @property
  def head(self):
    return self._head
  
  @property
  def node_type(self):
    return self._type
  
class RNNGAction(object):
  class Type(enum.Enum):
    GEN=0
    NT=1
    REDUCE_LEFT=2
    REDUCE_RIGHT=3
    REDUCE_NT=4
    NONE=5
    
  def __init__(self, action_type, action_content=None):
    self._action_type = action_type
    self._action_content = action_content
  
  @property
  def action_type(self):
    return self._action_type
  
  @property
  def action_content(self):
    return self._action_content
  
  @property
  def action_id(self):
    return self.action_type.value
  
  def str_token(self, surface_vocab, nt_vocab, edge_vocab):
    if self.action_type == self.Type.GEN:
      return "GEN('{}')".format(surface_vocab[self.action_content])
    elif self.action_type == self.Type.NT:
      return "NT('{}')".format(nt_vocab[self.action_content])
    elif self.action_type == self.Type.REDUCE_LEFT:
      return "RL('{}')".format(edge_vocab[self.action_content])
    elif self.action_type == self.Type.REDUCE_RIGHT:
      return "RR('{}')".format(edge_vocab[self.action_content])
    else:
      return "NONE()"
 
  def __eq__(self, other):
    if type(other) == type(self):
      return self.action_type == other.action_type and self.action_content == other.action_content
    else:
      return False

  
class DepTreeRNNGSequenceSentence(GraphSentence):
  def __init__(self,
               idx: Optional[numbers.Integral],
               score: Optional[numbers.Real],
               graph: HyperGraph = None,
               surface_vocab: Optional[Vocab] = None,
               nt_vocab: Optional[Vocab] = None,
               edge_vocab: Optional[Vocab] = None,
               all_surfaces: bool = False,
               num_padded: numbers.Integral = 0,
               actions: List[RNNGAction] = None,
               unpadded_sent: 'GraphSentence' = None) -> None:
    super().__init__(idx=idx,
                     score=score,
                     graph=graph,
                     value_vocab=surface_vocab,
                     node_vocab=nt_vocab,
                     edge_vocab=edge_vocab,
                     num_padded=num_padded,
                     unpadded_sent=unpadded_sent)

    assert graph is None or actions is None
    if actions is None:
      actions = self._actions_from_graph(all_surfaces)
    else:
      graph = self._graph_from_actions(actions)

    self.graph = graph
    self.actions = actions

  def sent_len(self) -> int:
    return len(self.actions)
  
  def len_unpadded(self) -> int:
    return len(self.actions)
  
  def str_tokens(self, **kwargs) -> List[str]:
    return [action.str_token(self.value_vocab, self.node_vocab, self.edge_vocab) for action in self.actions]

  def sent_str(self, custom_output_procs=None, **kwargs):
    return " ".join(self.str_tokens())

  def __getitem__(self, key: numbers.Integral) -> Optional[int]:
    return self.actions[key]

  def create_padded_sent(self, pad_len: numbers.Integral) -> 'GraphSentence':
    """
    Return padded graph.

    Args:
      pad_len: Number of tokens to pad.

    Returns:
      New padded graph, or self if pad_len==0.
    """
    padded_sent = copy.deepcopy(self)
    padded_sent.actions[-pad_len:] = [RNNGAction.Type.NONE for _ in range(pad_len)]
    return padded_sent
    
  def _actions_from_graph(self, all_surfaces):
    roots = self.graph.roots()
    # Only 1 Head
    assert len(roots) == 1
    # Helper function
    def actions_from_graph(current_id, results):
      successors = self.graph.sucessors(current_id, with_edge=True)
      for i in range(results[1], current_id+1):
        if len(successors) == 0 or all_surfaces: # Leaf
          results[0].append(RNNGAction(RNNGAction.Type.GEN, self.value_vocab.convert(self.graph[i].value)))
        else: # Non Terminal
          results[0].append(RNNGAction(RNNGAction.Type.NT, self.node_vocab.convert(self.graph[i].value)))
      results[1] = max(results[1], current_id+1)
      
      for child, edge in sorted(successors):
        actions_from_graph(child, results)
        if child < current_id:
          results[0].append(RNNGAction(RNNGAction.Type.REDUCE_LEFT, self.edge_vocab.convert(edge.label)))
        else:
          results[0].append(RNNGAction(RNNGAction.Type.REDUCE_RIGHT, self.edge_vocab.convert(edge.label)))
      return results[0]
    # Driver function
    return actions_from_graph(roots[0], [[], 1])
  
  def _graph_from_actions(self, actions):
    stack = []
    node_list = {}
    edge_list = []
    node_id = 0
    for action in actions:
      act_type = action.action_type
      if act_type == RNNGAction.Type.GEN or act_type == RNNGAction.Type.NT:
        new_node = HyperNode(action.action_content, node_id)
        stack.append(new_node)
        node_list[node_id] = new_node
        node_id += 1
      elif act_type == RNNGAction.Type.REDUCE_RIGHT:
        right = stack.pop()
        left = stack.pop()
        new_edge = HyperEdge(left.node_id, [right.node_id], None, action.action_content)
        edge_list.append(new_edge)
        stack.append(left)
      elif act_type == RNNGAction.Type.REDUCE_LEFT:
        right = stack.pop()
        left = stack.pop()
        new_edge = HyperEdge(right.node_id, [left.node_id], None, action.action_content)
        edge_list.append(new_edge)
        stack.append(right)
      elif act_type == RNNGAction.Type.REDUCE_NT:
        raise NotImplementedError("Finish this method for syntax tree generation!")
    return HyperGraph(edge_list, node_list)

class ParseTreeRNNGSequenceSentence(DepTreeRNNGSequenceSentence):
  def _actions_from_graph(self, all_surfaces):
    roots = self.graph.roots()
    assert len(roots) == 1
    
    def actions_from_graph(now_id, result, visited):
      now = self.graph[now_id]
      if now.node_id in visited:
        return
      visited.add(now.node_id)
      # Adding action accordingly
      if now.type == SyntaxTreeNode.Type.NT or now.type == SyntaxTreeNode.Type.PRT:
        result.append(RNNGAction(RNNGAction.Type.NT, now.head))
        # Enqueue next action
        for child_id in reversed(self.graph.sucessors(now.node_id)):
          actions_from_graph(child_id, result, visited)
        result.append(RNNGAction(RNNGAction.Type.REDUCE_NT))
      else:
        result.append(RNNGAction(RNNGAction.Type.GEN, now.head))
    
    buffer_result = []
    actions_from_graph(roots[0], buffer_result, set())
    return buffer_result
    
