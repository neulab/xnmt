import unittest
from xnmt.input_readers import CoNLLTreeReader

class TestDependencyTree(unittest.TestCase):
  def setUp(self):
    if not hasattr(self, "trees"):
      self.trees = CoNLLTreeReader().read_sents("examples/data/dep_tree.conll")
  
  def test_read_file(self):
    pass
  
  
if __name__ == "__main__":
  unittest.main