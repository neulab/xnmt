import unittest

from xnmt.graph import HyperGraph, HyperEdge, HyperNode


class TestGraph(unittest.TestCase):
  def setUp(self):
    nodes = [HyperNode('a', 1), HyperNode('b', 2), HyperNode('c', 3), HyperNode('d', 4), HyperNode('e', 5)]
    
    edg_list = [HyperEdge(nodes[0], [nodes[1]]),
                HyperEdge(nodes[0], [nodes[2]]),
                HyperEdge(nodes[1], [nodes[3]]),
                HyperEdge(nodes[1], [nodes[4]])]
    self.nodes = nodes
    self.graph = HyperGraph(edg_list)
    
  def test_construction(self):
    pred_list = {2: [1], 3: [1], 4: [2], 5: [2]}
    adj_list = {1: [2, 3], 2: [4, 5]}
    node_list = {1: self.nodes[0], 2: self.nodes[1], 3: self.nodes[2], 4: self.nodes[3], 5: self.nodes[4]}
    self.assertDictEqual(adj_list, self.graph._succ_list)
    self.assertDictEqual(pred_list, self.graph._pred_list)
    self.assertDictEqual(node_list, self.graph._node_list)
    
  def test_reverse(self):
    reverse_graph = self.graph.reverse()
    # Now we reverse the expected adj_list & predec list
    pred_list = {1: [2, 3], 2: [4, 5]}
    adj_list = {2: [1], 3: [1], 4: [2], 5: [2]}
    node_list = {1: self.nodes[0], 2: self.nodes[1], 3: self.nodes[2], 4: self.nodes[3], 5: self.nodes[4]}
    self.assertDictEqual(adj_list, reverse_graph._succ_list)
    self.assertDictEqual(pred_list, reverse_graph._pred_list)
    self.assertDictEqual(node_list, reverse_graph._node_list)
  
  def test_toposort(self):
    # Taken from https://www.geeksforgeeks.org/topological-sorting/
    nodes = []
    for i in range(6):
      nodes.append(HyperNode(i, i))
    edges = [HyperEdge(nodes[5], [nodes[2], nodes[0]]),
             HyperEdge(nodes[4], [nodes[0], nodes[1]]),
             HyperEdge(nodes[2], [nodes[3]]),
             HyperEdge(nodes[3], [nodes[1]])]
    graph = HyperGraph(edges)
    self.assertListEqual(graph.topo_sort(), [5, 4, 2, 3, 1, 0])
    
  def test_leaves(self):
    self.assertListEqual(self.graph.leaves(), [3, 4, 5])


if __name__ == "__main__":
  unittest.main