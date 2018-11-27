from typing import List, Dict
import numbers


class Node(object):
  """
  Represents a single (hyper)-node in a graph
  """
  def __init__(self,
               data: str = None,
               node_type: str = None,
               node_id: numbers.Integral = None,
               edges = None):
    self.data = data
    self.node_type = node_type
    self.node_id = node_id
    self.edges = edges or []
    
  def __repr__(self):
    if len(self.edges) != 0:
      children = " -> {}".format(" ".join([repr(edge) for edge in self.edges]))
    else:
      children = ""
    
    return "Node {:d} ({}:{}){}".format(self.node_id, self.data,
                                        self.node_type, children)


class Edge(object):
  """
  Represents an (hyper)-edge in a graph
  """
  def __init__(self,
               node_from:Node,
               node_to:List[Node],
               weight:numbers.Integral = 0,
               name:str = None,
               edge_id:numbers.Integral = -1):
    self.node_from = node_from
    self.node_to = node_to
    self.weight = weight
    self.name = name
    self.edge_id = edge_id
  
  def __repr__(self):
    return "[{:s}, ({:s})]".format(self.name, ", ".join([str(node.node_id) for node in self.node_to]))


class Graph(object):
  """
  Represents the graph data structure
  """
  def __init__(self, edg_list : List[Edge], nodes: Dict[numbers.Integral, Node], root: Node):
    self.edg_list = edg_list
    self.nodes = nodes
    self.root = root
    
  def __repr__(self):
    representation = []
    for node in sorted(self.nodes.values(), key=lambda node: node.node_id):
      representation.append(repr(node))
    return "\n".join(representation)


class DependencyTree(Graph):
  @staticmethod
  def from_conll(conll_line):
    nodes = {}
    edges = []
    def get_or_insert(node_id):
      if node_id not in nodes:
        nodes[node_id] = Node(None, node_id=node_id)
      return nodes[node_id]
 
    for node_id, form, lemma, pos, feat, head, deprel in conll_line:
      node_id, head_id = int(node_id), int(head)
      node_from = get_or_insert(head_id)
      node_to = get_or_insert(node_id)
      node_to.data = form
      node_to.node_type = pos
      
      edge = Edge(node_from, [node_to], name=deprel, edge_id=node_id)
      node_from.edges.append(edge)
      nodes[node_id] = node_to
      nodes[head_id] = node_from
      edges.append(edge)
    root = nodes[0]
    root.node_type = 0
    root.data = 0
    return DependencyTree(edges, nodes, root)
 