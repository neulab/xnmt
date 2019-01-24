from typing import List, Any
from collections import defaultdict

import functools


class HyperNode(object):
  """
  Represents a single HyperNode in a graph.
  - data: Value of the node
  - node_id: A unique id of the node.
  """
  def __init__(self, data: Any, node_id: int):
    self._data = data
    self._node_id = node_id
    
  @property
  def data(self):
    return self._data
  
  @property
  def node_id(self):
    return self._node_id
  
  def __repr__(self):
    return "Node({}, {})".format(self.node_id, str(self.data))


class HyperEdge(object):
  """
  Represents a single HyperEdge in a graph.
  - node_from: Source Node.
  - node_to: Destintation Nodes.
  - features: Float values representing the weight/features of the weight.
  """
  def __init__(self,
               node_from: HyperNode,
               node_to: List[HyperNode],
               features: List[float] = None):
    self._node_from = node_from
    self._node_to = tuple(node_to)
    self._features = tuple(features) if features is not None else features
    
  @property
  def node_from(self):
    return self._node_from

  @property
  def node_to(self):
    return self._node_to
  
  @property
  def features(self):
    return self._features
  
  def __repr__(self):
    return "Edge({} -> {}, {})".format(self.node_from.node_id,
                                   str([child.node_id for child in self.node_to]),
                                   str(self.features))


class HyperGraph(object):
  """
  A hypergraph datastructure. Represented with a list of HyperEdge.
  - edge_list: The list of hyperedge forming the graph.
  """
  def __init__(self, edge_list: List[HyperEdge]):
    self._edge_list = tuple(edge_list)
    adj_list, pred_list, node_list = self._build_graph()
    self._adj_list = adj_list
    self._pred_list = pred_list
    self._node_list = node_list

  # If hypergraph is immutable, we can cache the reverse of the graph
  @functools.lru_cache(maxsize=1)
  def reverse(self):
    rev_edge_list = []
    for edge in self._edge_list:
      assert len(edge.node_to) == 1, "Does not support reversed of HyperGraph yet."
      rev_edge_list.append(HyperEdge(edge.node_to[0], [edge.node_from], edge.features))
    return HyperGraph(rev_edge_list)
  
  # If hypergraph is immutable, we can cache the topological sort of the graph
  @functools.lru_cache(maxsize=1)
  def topo_sort(self):
    stack = []
    visited = [False for _ in range(len(self._node_list))]
    for node_id in sorted(self._node_list.keys()):
      if not visited[node_id]:
        self._topo_sort(node_id, visited, stack)
    return [self._node_list[node_id] for node_id in stack]
    
  def _topo_sort(self, node_id, visited, stack):
    visited[node_id] = True
    if node_id in self._adj_list:
      for adj_id in self._adj_list[node_id]:
        if not visited[adj_id]:
          self._topo_sort(adj_id, visited, stack)
    stack.append(node_id)
  
  def _build_graph(self):
    pred_list = defaultdict(list)
    adj_list = defaultdict(list)
    node_list = {}
    for edge in self._edge_list:
      from_id = edge.node_from.node_id
      for dest in edge.node_to:
        to_id = dest.node_id
        adj_list[from_id].append(to_id)
        pred_list[to_id].append(from_id)
        node_list[from_id] = edge.node_from
        node_list[to_id] = dest
    return dict(pred_list), dict(adj_list), node_list
 
  def __repr__(self):
    lst = []
    for node in self._node_list.values():
      lst.append(repr(node))
    for edge in self._edge_list:
      lst.append(repr(edge))
    return "\n".join(lst)
