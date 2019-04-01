import argparse
import xnmt.input_readers as input_readers

from xnmt.graph import HyperNode, HyperEdge, HyperGraph
from xnmt.sent import SyntaxTreeNode
from collections import defaultdict

DELIMITER = "▁"

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("conll_tree_input")
parser.add_argument("sp_input")
args = parser.parse_args()

class DummyVocab:
  def __init__(self):
    self.vocab = defaultdict(lambda: len(self.vocab))
    self.backvocab = {}
    
  def convert(self, word):
    wordid = self.vocab[word]
    self.backvocab[wordid] = word
    return wordid
    
  def __getitem__(self, item):
    return self.backvocab[item]
  
# Initialization
surface_vocab = DummyVocab()
nt_vocab = DummyVocab()
edge_vocab = DummyVocab()
reader = input_readers.CoNLLToRNNGActionsReader(surface_vocab=surface_vocab,
                                                nt_vocab=nt_vocab,
                                                edg_vocab=edge_vocab)

# Reading Input
input_tree = list(reader.read_sents(args.conll_tree_input))
with open(args.sp_input) as fp:
  input_sp= fp.readlines()
  
assert len(input_tree) == len(input_sp)

def graph_to_conll(graph):
  ret = []
  for node in graph.iter_nodes():
    # 1	can	_	MD	_	3	aux
    pred = graph.predecessors(node.node_id, True)
    if len(pred) == 0:
      ret.append("{}\t{}\t_\t{}\t_\t{}\t{}".format(node.node_id, node.value, node.head, 0, "ROOT"))
    else:
      pred_id, pred_edge = pred[0]
      ret.append("{}\t{}\t_\t{}\t_\t{}\t{}".format(node.node_id, node.value, node.head, pred_id, pred_edge.label))
  return "\n".join(ret)
  
def remap_id(node_list, edge_list, leaves):
  id_mapping = {}
  for i, node in enumerate(leaves):
    id_mapping[node.node_id] = i+1
  # New edge + node with new id mapping
  out_node_list = {}
  out_edge_list = []
  for node_id, node in node_list.items():
    out_node_list[id_mapping[node_id]] = SyntaxTreeNode(id_mapping[node_id], node.value, node.head, node.node_type)
  for edge in edge_list:
    out_edge_list.append(HyperEdge(id_mapping[edge.node_from],
                                   [id_mapping[edge.node_to[0]]],
                                   edge.features,
                                   edge.label))
  return HyperGraph(out_edge_list, out_node_list)

def normalize_space_at_conll(tree):
  graph = tree.graph
  leaves = []
  node_list = {}
  edge_list = []
  now_id = graph.len_nodes
  for edge in graph.iter_edges():
    edge_list.append(edge)
  edge_list = edge_list[:-1]
  for i in range(1, graph.len_nodes):
    node = graph[i]
    word = node.value
    for j, subword in enumerate(word.split()):
      if j == 0:
        node_list[node.node_id] = SyntaxTreeNode(node.node_id, subword, node.head, node.node_type)
        leaves.append(node_list[node.node_id])
      else:
        node_list[now_id] = SyntaxTreeNode(now_id, subword, node.head, node.node_type)
        leaves.append(node_list[now_id])
        edge_list.append(HyperEdge(node.node_id, [now_id], None, "[whtsp]"))
        now_id += 1
  return remap_id(node_list, edge_list, leaves)

# Processing
def normalize_sentpiece(sp, graph):
  node_list = {}
  edge_list = []
  leaves = []
  sp = sp.strip().split() + [DELIMITER]
  # Create new hyperedge list
  for edge in graph.iter_edges():
    edge_list.append(edge)
  # Routine to modift the graph structure
  def write_changes(buffer, idx, now_id):
    now_node = graph[idx]
    node_list[idx] = SyntaxTreeNode(now_node.node_id, buffer[0], now_node.head, now_node.node_type)
    leaves.append(node_list[idx])
    for i in range(1, len(buffer)):
      node_list[now_id] = SyntaxTreeNode(now_id, buffer[i], "["+ now_node.head + "]", now_node.node_type)
      edge_list.append(HyperEdge(idx, [now_id], edge.features, "[sp]"))
      leaves.append(node_list[now_id])
      now_id += 1
    return now_id
  # Synchronously modify the sentpiece and tree
  idx = 0
  buffer = []
  now_id = graph.len_nodes + 1
  for token in sp:
    if DELIMITER in token:
      if idx != 0:
        now_id = write_changes(buffer, idx, now_id)
        buffer.clear()
      idx += 1
    buffer.append(token)
  return remap_id(node_list, edge_list, leaves)

for sp, tree in zip(input_sp, input_tree):
  print(graph_to_conll(normalize_sentpiece(sp, normalize_space_at_conll(tree))))
  print()
