import unittest
import copy

import yaml

import xnmt
from xnmt import util, persistence
from xnmt.persistence import Path, YamlPreloader, Serializable, serializable_init, bare

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
    s = {Path("one.2"), Path("one.1.3"), Path("one.1.3")}
    self.assertIn(Path("one.2"), s)
    self.assertEqual(len(s), 2)
  def test_append(self):
    self.assertEqual(str(Path("one").append("2")), "one.2")
    self.assertEqual(str(Path("").append("2")), "2")
    self.assertEqual(str(Path(".").append("2")), ".2")
    self.assertEqual(str(Path(".1.2").append("2")), ".1.2.2")
    with self.assertRaises(ValueError):
      Path("one").append("")
    with self.assertRaises(ValueError):
      Path("one").append(".")
    with self.assertRaises(ValueError):
      Path("one").append("two.3")
  def test_add_path(self):
    self.assertEqual(str(Path("one").add_path(Path("2"))), "one.2")
    self.assertEqual(str(Path("one").add_path(Path("2.3"))), "one.2.3")
    self.assertEqual(str(Path("").add_path(Path("2.3"))), "2.3")
    self.assertEqual(str(Path("one.2").add_path(Path(""))), "one.2")
    self.assertEqual(str(Path("").add_path(Path(""))), "")
    self.assertEqual(str(Path(".").add_path(Path(""))), ".")
    self.assertEqual(str(Path(".").add_path(Path("one.two"))), ".one.two")
    self.assertEqual(str(Path(".xy").add_path(Path("one.two"))), ".xy.one.two")
    with self.assertRaises(NotImplementedError):
      Path("one").add_path(Path(".2.3"))
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
  def test_get_item_slice(self):
    self.assertEqual(str(Path("one")[0:1]), "one")
    self.assertEqual(str(Path("one.2.3")[1:3]), "2.3")
    self.assertEqual(str(Path("one.2.3")[0:-1]), "one.2")
    self.assertEqual(str(Path("one.2.3")[-1:]), "3")
    with self.assertRaises(ValueError):
      Path(".one.2.3")[0:1:-1]
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
    self.assertEqual(Path("").ancestors(), {Path("")})
    self.assertEqual(Path("a").ancestors(), {Path(""), Path("a")})
    self.assertEqual(Path("one.two.three").ancestors(), {Path(""), Path("one"), Path("one.two"), Path("one.two.three")})

class DummyClass(Serializable):
  yaml_tag = "!DummyClass"
  @serializable_init
  def __init__(self, arg1, arg2="{V2}", arg3="{V3}"):
    self.arg1 = arg1
    self.arg2 = arg2
    self.arg3 = arg3
class DummyClass2(Serializable):
  yaml_tag = "!DummyClass2"
  @serializable_init
  def __init__(self, arg1=bare(DummyClass)):
    self.arg1 = arg1
class DummyClass3(Serializable):
  yaml_tag = "!DummyClass3"
  @serializable_init
  def __init__(self, arg1=bare(DummyClass2)):
    self.arg1 = arg1
class DummyClassForgotBare(Serializable):
  yaml_tag = "!DummyClassForgotBare"
  @serializable_init
  def __init__(self, arg1=DummyClass("")):
    self.arg1 = arg1

class TestPreloader(unittest.TestCase):
  def setUp(self):
    yaml.add_representer(DummyClass, xnmt.init_representer)
    self.out_dir = "test/tmp"
    util.make_parent_dir(f"{self.out_dir}/asdf")

  def test_experiment_names_from_file(self):
    with open(f"{self.out_dir}/tmp.yaml", "w") as f_out:
      yaml.dump({
          "exp1": DummyClass(""),
          "exp2": DummyClass(""),
          "exp10": DummyClass("")
        },
        f_out)
    self.assertListEqual(YamlPreloader.experiment_names_from_file(f"{self.out_dir}/tmp.yaml"),
                         ["exp1", "exp10", "exp2"])

  def test_inconsistent_loadserialized(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"""
    a: !LoadSerialized
      filename: {self.out_dir}/tmp1.yaml
      bad_arg: 1
    """)
    with self.assertRaises(ValueError):
      YamlPreloader.preload_obj(test_obj, "SOME_EXP_NAME", "SOME_EXP_DIR")
  def test_inconsistent_loadserialized(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"""
    a: !LoadSerialized
      filename: {self.out_dir}/tmp1.yaml
      overwrite:
      - path: a
      - val: b
    """)
    with self.assertRaises(ValueError):
      YamlPreloader.preload_obj(test_obj, "SOME_EXP_NAME", "SOME_EXP_DIR")

  def test_placeholder_loadserialized(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"""
    a: !LoadSerialized
      filename: '{{EXP_DIR}}/{{EXP}}.yaml'
    """)
    YamlPreloader.preload_obj(test_obj, exp_name = "tmp1", exp_dir=self.out_dir)

  def test_load_referenced_serialized_top(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"!LoadSerialized {{ filename: {self.out_dir}/tmp1.yaml }}")
    loaded_obj = YamlPreloader._load_referenced_serialized(test_obj)
    self.assertIsInstance(loaded_obj, DummyClass)
    self.assertEqual(loaded_obj.arg1, "v1")

  def test_load_referenced_serialized_nested(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"""
    a: 1
    b: !LoadSerialized
      filename: {self.out_dir}/tmp1.yaml
      overwrite:
      - path: arg1
        val: !LoadSerialized
              filename: {self.out_dir}/tmp1.yaml
    """)
    loaded_obj = YamlPreloader._load_referenced_serialized(test_obj)
    self.assertIsInstance(loaded_obj["b"], DummyClass)
    self.assertIsInstance(loaded_obj["b"].arg1, DummyClass)

  def test_resolve_kwargs(self):
    test_obj = yaml.load("""
    !DummyClass
      kwargs:
        arg1: 1
        other_arg: 2
    """)
    YamlPreloader._resolve_kwargs(test_obj)
    self.assertFalse(hasattr(test_obj, "kwargs"))
    self.assertFalse(hasattr(test_obj, "arg2"))
    self.assertEqual(getattr(test_obj, "arg1", None), 1)
    self.assertEqual(getattr(test_obj, "other_arg", None), 2)

  def test_resolve_bare_default_args(self):
    test_obj = yaml.load("""
                         a: !DummyClass
                           arg1: !DummyClass2 {}
                         b: !DummyClass3 {}
                         """)
    YamlPreloader._resolve_bare_default_args(test_obj)
    self.assertIsInstance(test_obj["a"].arg1.arg1, DummyClass)
    self.assertIsInstance(test_obj["b"].arg1, DummyClass2)
    self.assertIsInstance(test_obj["b"].arg1.arg1, DummyClass)

  def test_resolve_bare_default_args_illegal(self):
    test_obj = yaml.load("""
                         a: !DummyClassForgotBare {}
                         """)
    with self.assertRaises(ValueError):
      YamlPreloader._resolve_bare_default_args(test_obj)

  def test_format_strings(self):
    test_obj = yaml.load("""
                         a: !DummyClass
                           arg1: '{V1}'
                           other_arg: 2
                         b: !DummyClass
                           arg1: 1
                           other_arg: '{V2}'
                         c: '{V1}/bla'
                         d: ['bla', 'bla.{V2}']
                         """)
    YamlPreloader._format_strings(test_obj, {"V1":"val1", "V2":"val2"})
    self.assertEqual(test_obj["a"].arg1, "val1")
    self.assertEqual(test_obj["a"].other_arg, 2)
    self.assertEqual(test_obj["a"].arg2, "val2")
    self.assertFalse(hasattr(test_obj["a"], "arg3"))
    self.assertEqual(test_obj["b"].arg1, 1)
    self.assertEqual(test_obj["b"].other_arg, '{V2}')
    self.assertEqual(test_obj["b"].arg2, "val2")
    self.assertFalse(hasattr(test_obj["b"], "arg3"))
    self.assertEqual(test_obj["c"], "val1/bla")
    self.assertListEqual(test_obj["d"], ["bla", "bla.val2"])

if __name__ == '__main__':
  unittest.main()
