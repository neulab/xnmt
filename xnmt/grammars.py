from typing import List

class RNNGAction(object):
  pass


class Gen(RNNGAction):
  def __init__(self, word_id):
    self.word_id = word_id
  def __repr__(self):
    return "GEN({})".format(self.word_id)

    
class NT(RNNGAction):
  def __init__(self, head_id):
    self.head_id = head_id
  def __repr__(self):
    return "NT({})".format(self.head_id)


class Reduce(RNNGAction):
  def __init__(self, is_left, label=None):
    self.is_left = is_left
    self.label = label
  def __repr__(self):
    if self.is_left:
      return "RL({})".format(self.label)
    else:
      return "RR({})".format(self.label)


class ShiftReduceGrammar(object):
  """
  Generative RNNG
  """
  def __init__(self, actions: List[RNNGAction]):
    self.actions = actions
    
  @staticmethod
  def from_graph(graph, no_nt=True):
    nodes = graph.nodes
    results = [[], 0]
    ShiftReduceGrammar.from_graph_rec(nodes[0], results, nodes, no_nt)
    return ShiftReduceGrammar(actions=results[0])

  @staticmethod
  def from_graph_rec(current, results, nodes, no_nt):
    for i in range(results[1], current.node_id+1):
      if no_nt or len(nodes[i].edges) == 0:
        results[0].append(Gen(nodes[i].data))
      else:
        results[0].append(NT(nodes[i].data))
    results[1] = max(results[1], current.node_id+1)
    
    for edge in current.edges:
      for child in edge.node_to:
        ShiftReduceGrammar.from_graph_rec(child, results, nodes, no_nt)
        results[0].append(Reduce(child.node_id < current.node_id, edge.name))

