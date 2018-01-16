import unittest

from xnmt.serialize.tree_tools import Path

class TestPath(unittest.TestCase):

  def setUp(self):
    pass

  def test_init(self):
    self.assertTrue(type(Path(""))==Path)
    self.assertTrue(type(Path(".."))==Path)
    self.assertTrue(type(Path(".2"))==Path)
    self.assertTrue(type(Path("one.2"))==Path)
    with self.assertRaises(ValueError):
      Path(".one.")
      Path("one..2")
  def test_str(self):
    self.assertEqual(str(Path("one.2")), "one.2")
    self.assertEqual(str(Path("")), "")
  def test_set(self):
    s = set([Path("one.2"), Path("one.1.3"), Path("one.1.3")])
    self.assertIn(Path("one.2"), s)
    self.assertEquals(len(s), 2)
  def test_append(self):
    self.assertEqual(str(Path("one").append("2")), "one.2")
    self.assertEqual(str(Path("").append("2")), "2")
    self.assertEqual(str(Path(".").append("2")), ".2")
    with self.assertRaises(ValueError):
      Path("one").append("")
    with self.assertRaises(ValueError):
      Path("one").append(".")
    with self.assertRaises(ValueError):
      Path("one").append("two.3")
  def test_get_absolute(self):
    self.assertEqual(Path(".").get_absolute(Path("1.2")), Path("1.2"))
    self.assertEqual(Path(".x.y").get_absolute(Path("1.2")), Path("1.2.x.y"))
    self.assertEqual(Path("..x.y").get_absolute(Path("1.2")), Path("1.x.y"))
    self.assertEqual(Path("...x.y").get_absolute(Path("1.2")), Path("x.y"))
    with self.assertRaises(ValueError):
      Path("....x.y").get_absolute(Path("1.2"))
  def test_descend_one(self):
    self.assertEqual(str(Path("one.2.3").descend_one()), "2.3")
    self.assertEqual(str(Path("3").descend_one()), "")
    with self.assertRaises(ValueError):
      Path("").descend_one()
    with self.assertRaises(ValueError):
      Path(".one.2").descend_one()
  def test_len(self):
    self.assertEqual(len(Path("")), 0)
    self.assertEqual(len(Path("one")), 1)
    self.assertEqual(len(Path("one.2.3")), 3)
    with self.assertRaises(ValueError):
      len(Path(".one"))
      len(Path("."))
  def test_get_item(self):
    self.assertEqual(Path("one")[0], "one")
    self.assertEqual(Path("one.2.3")[0], "one")
    self.assertEqual(Path("one.2.3")[2], "3")
    self.assertEqual(Path("one.2.3")[-1], "3")
    with self.assertRaises(ValueError):
      Path(".one.2.3")[-1]
  def test_parent(self):
    self.assertEqual(Path("one").parent(), Path(""))
    self.assertEqual(Path("one.two.three").parent(), Path("one.two"))
    self.assertEqual(Path(".one").parent(), Path("."))
    with self.assertRaises(ValueError):
      Path(".").parent()
    with self.assertRaises(ValueError):
      Path("").parent()
  def test_eq(self):
    self.assertEqual(Path(""), Path(""))
    self.assertEqual(Path(".."), Path(".."))
    self.assertEqual(Path("one.2"), Path("one.2"))
    self.assertEqual(Path("one.2"), Path("one.2.3").parent())
    self.assertNotEqual(Path("one.2"), Path("one.2.3"))
    self.assertNotEqual(Path(""), Path("."))
  def test_ancestors(self):
    self.assertEqual(Path("").ancestors(), set([Path("")]))
    self.assertEqual(Path("a").ancestors(), set([Path(""),Path("a")]))
    self.assertEqual(Path("one.two.three").ancestors(), set([Path(""), Path("one"), Path("one.two"), Path("one.two.three")]))

if __name__ == '__main__':
  unittest.main()
