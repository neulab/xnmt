import unittest

from xnmt.graph import HyperGraph, HyperEdge, HyperNode


class TestGraph(unittest.TestCase):
  def setUp(self):
    nodes = {1:HyperNode('a', 1),
             2:HyperNode('b', 2),
             3:HyperNode('c', 3),
             4:HyperNode('d', 4),
             5:HyperNode('e', 5)}
    
    edg_list = [HyperEdge(1, [2]),
                HyperEdge(1, [3]),
                HyperEdge(2, [4]),
                HyperEdge(2, [5])]
    self.nodes = nodes
    self.graph = HyperGraph(edg_list, nodes)
    
  def test_construction(self):
    pred_list = {2: [1], 3: [1], 4: [2], 5: [2]}
    adj_list = {1: [2, 3], 2: [4, 5]}
    self.assertDictEqual(adj_list, self.graph._succ_list)
    self.assertDictEqual(pred_list, self.graph._pred_list)
    
  def test_reverse(self):
    reverse_graph = self.graph.reverse()
    # Now we reverse the expected adj_list & predec list
    pred_list = {1: [2, 3], 2: [4, 5]}
    adj_list = {2: [1], 3: [1], 4: [2], 5: [2]}
    self.assertDictEqual(adj_list, reverse_graph._succ_list)
    self.assertDictEqual(pred_list, reverse_graph._pred_list)
  
  def test_toposort(self):
    # Taken from https://www.geeksforgeeks.org/topological-sorting/
    nodes = {}
    for i in range(6):
      nodes[i] = HyperNode(i, i)
    edges = [HyperEdge(5, [2, 0]),
             HyperEdge(4, [0, 1]),
             HyperEdge(2, [3]),
             HyperEdge(3, [1])]
    graph = HyperGraph(edges, nodes)
    self.assertListEqual(graph.topo_sort(), [5, 4, 2, 3, 1, 0])
    
  def test_leaves(self):
    self.assertListEqual(self.graph.leaves(), [3, 4, 5])


if __name__ == "__main__":
  unittest.main