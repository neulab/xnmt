import unittest
import os
import shutil

import yaml

import xnmt
from xnmt import events, param_collections, persistence, utils

class TestPath(unittest.TestCase):

  def setUp(self):
    pass

  def test_init(self):
    self.assertTrue(type(persistence.Path(""))==persistence.Path)
    self.assertTrue(type(persistence.Path(".."))==persistence.Path)
    self.assertTrue(type(persistence.Path(".2"))==persistence.Path)
    self.assertTrue(type(persistence.Path("one.2"))==persistence.Path)
    with self.assertRaises(ValueError):
      persistence.Path(".one.")
      persistence.Path("one..2")
  def test_str(self):
    self.assertEqual(str(persistence.Path("one.2")), "one.2")
    self.assertEqual(str(persistence.Path("")), "")
  def test_set(self):
    s = {persistence.Path("one.2"), persistence.Path("one.1.3"), persistence.Path("one.1.3")}
    self.assertIn(persistence.Path("one.2"), s)
    self.assertEqual(len(s), 2)
  def test_append(self):
    self.assertEqual(str(persistence.Path("one").append("2")), "one.2")
    self.assertEqual(str(persistence.Path("").append("2")), "2")
    self.assertEqual(str(persistence.Path(".").append("2")), ".2")
    self.assertEqual(str(persistence.Path(".1.2").append("2")), ".1.2.2")
    with self.assertRaises(ValueError):
      persistence.Path("one").append("")
    with self.assertRaises(ValueError):
      persistence.Path("one").append(".")
    with self.assertRaises(ValueError):
      persistence.Path("one").append("two.3")
  def test_add_path(self):
    self.assertEqual(str(persistence.Path("one").add_path(persistence.Path("2"))), "one.2")
    self.assertEqual(str(persistence.Path("one").add_path(persistence.Path("2.3"))), "one.2.3")
    self.assertEqual(str(persistence.Path("").add_path(persistence.Path("2.3"))), "2.3")
    self.assertEqual(str(persistence.Path("one.2").add_path(persistence.Path(""))), "one.2")
    self.assertEqual(str(persistence.Path("").add_path(persistence.Path(""))), "")
    self.assertEqual(str(persistence.Path(".").add_path(persistence.Path(""))), ".")
    self.assertEqual(str(persistence.Path(".").add_path(persistence.Path("one.two"))), ".one.two")
    self.assertEqual(str(persistence.Path(".xy").add_path(persistence.Path("one.two"))), ".xy.one.two")
    with self.assertRaises(NotImplementedError):
      persistence.Path("one").add_path(persistence.Path(".2.3"))
  def test_get_absolute(self):
    self.assertEqual(persistence.Path(".").get_absolute(persistence.Path("1.2")), persistence.Path("1.2"))
    self.assertEqual(persistence.Path(".x.y").get_absolute(persistence.Path("1.2")), persistence.Path("1.2.x.y"))
    self.assertEqual(persistence.Path("..x.y").get_absolute(persistence.Path("1.2")), persistence.Path("1.x.y"))
    self.assertEqual(persistence.Path("...x.y").get_absolute(persistence.Path("1.2")), persistence.Path("x.y"))
    with self.assertRaises(ValueError):
      persistence.Path("....x.y").get_absolute(persistence.Path("1.2"))
  def test_descend_one(self):
    self.assertEqual(str(persistence.Path("one.2.3").descend_one()), "2.3")
    self.assertEqual(str(persistence.Path("3").descend_one()), "")
    with self.assertRaises(ValueError):
      persistence.Path("").descend_one()
    with self.assertRaises(ValueError):
      persistence.Path(".one.2").descend_one()
  def test_len(self):
    self.assertEqual(len(persistence.Path("")), 0)
    self.assertEqual(len(persistence.Path("one")), 1)
    self.assertEqual(len(persistence.Path("one.2.3")), 3)
    with self.assertRaises(ValueError):
      len(persistence.Path(".one"))
      len(persistence.Path("."))
  def test_get_item(self):
    self.assertEqual(persistence.Path("one")[0], "one")
    self.assertEqual(persistence.Path("one.2.3")[0], "one")
    self.assertEqual(persistence.Path("one.2.3")[2], "3")
    self.assertEqual(persistence.Path("one.2.3")[-1], "3")
    with self.assertRaises(ValueError):
      persistence.Path(".one.2.3")[-1]
  def test_get_item_slice(self):
    self.assertEqual(str(persistence.Path("one")[0:1]), "one")
    self.assertEqual(str(persistence.Path("one.2.3")[1:3]), "2.3")
    self.assertEqual(str(persistence.Path("one.2.3")[0:-1]), "one.2")
    self.assertEqual(str(persistence.Path("one.2.3")[-1:]), "3")
    with self.assertRaises(ValueError):
      persistence.Path(".one.2.3")[0:1:-1]
  def test_parent(self):
    self.assertEqual(persistence.Path("one").parent(), persistence.Path(""))
    self.assertEqual(persistence.Path("one.two.three").parent(), persistence.Path("one.two"))
    self.assertEqual(persistence.Path(".one").parent(), persistence.Path("."))
    with self.assertRaises(ValueError):
      persistence.Path(".").parent()
    with self.assertRaises(ValueError):
      persistence.Path("").parent()
  def test_eq(self):
    self.assertEqual(persistence.Path(""), persistence.Path(""))
    self.assertEqual(persistence.Path(".."), persistence.Path(".."))
    self.assertEqual(persistence.Path("one.2"), persistence.Path("one.2"))
    self.assertEqual(persistence.Path("one.2"), persistence.Path("one.2.3").parent())
    self.assertNotEqual(persistence.Path("one.2"), persistence.Path("one.2.3"))
    self.assertNotEqual(persistence.Path(""), persistence.Path("."))
  def test_ancestors(self):
    self.assertEqual(persistence.Path("").ancestors(), {persistence.Path("")})
    self.assertEqual(persistence.Path("a").ancestors(), {persistence.Path(""), persistence.Path("a")})
    self.assertEqual(persistence.Path("one.two.three").ancestors(), {persistence.Path(""), persistence.Path("one"), persistence.Path("one.two"), persistence.Path("one.two.three")})

class DummyClass(persistence.Serializable):
  yaml_tag = "!DummyClass"
  @persistence.serializable_init
  def __init__(self, arg1, arg2="{V2}", arg3="{V3}"):
    self.arg1 = arg1
    self.arg2 = arg2
    self.arg3 = arg3
class DummyClass2(persistence.Serializable):
  yaml_tag = "!DummyClass2"
  @persistence.serializable_init
  def __init__(self, arg1=persistence.bare(DummyClass)):
    self.arg1 = arg1
class DummyClass3(persistence.Serializable):
  yaml_tag = "!DummyClass3"
  @persistence.serializable_init
  def __init__(self, arg1=persistence.bare(DummyClass2)):
    self.arg1 = arg1
class DummyClassForgotBare(persistence.Serializable):
  yaml_tag = "!DummyClassForgotBare"
  @persistence.serializable_init
  def __init__(self, arg1=DummyClass("")):
    self.arg1 = arg1

class TestPreloader(unittest.TestCase):
  def setUp(self):
    yaml.add_representer(DummyClass, xnmt.init_representer)
    self.out_dir = "test/tmp"
    utils.make_parent_dir(f"{self.out_dir}/asdf")

  def test_experiment_names_from_file(self):
    with open(f"{self.out_dir}/tmp.yaml", "w") as f_out:
      yaml.dump({
          "exp1": DummyClass(""),
          "exp2": DummyClass(""),
          "exp10": DummyClass("")
        },
        f_out)
    self.assertListEqual(persistence.YamlPreloader.experiment_names_from_file(f"{self.out_dir}/tmp.yaml"),
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
      persistence.YamlPreloader.preload_obj(test_obj, "SOME_EXP_NAME", "SOME_EXP_DIR")
  def test_inconsistent_loadserialized2(self):
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
      persistence.YamlPreloader.preload_obj(test_obj, "SOME_EXP_NAME", "SOME_EXP_DIR")

  def test_placeholder_loadserialized(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"""
    a: !LoadSerialized
      filename: '{{EXP_DIR}}/{{EXP}}.yaml'
    """)
    persistence.YamlPreloader.preload_obj(test_obj, exp_name = "tmp1", exp_dir=self.out_dir)

  def test_load_referenced_serialized_top(self):
    with open(f"{self.out_dir}/tmp1.yaml", "w") as f_out:
      yaml.dump(DummyClass(arg1="v1"), f_out)
    test_obj = yaml.load(f"!LoadSerialized {{ filename: {self.out_dir}/tmp1.yaml }}")
    loaded_obj = persistence.YamlPreloader._load_serialized(test_obj)
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
    loaded_obj = persistence.YamlPreloader._load_serialized(test_obj)
    self.assertIsInstance(loaded_obj["b"], DummyClass)
    self.assertIsInstance(loaded_obj["b"].arg1, DummyClass)

  def test_resolve_kwargs(self):
    test_obj = yaml.load("""
    !DummyClass
      kwargs:
        arg1: 1
        other_arg: 2
    """)
    persistence.YamlPreloader._resolve_kwargs(test_obj)
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
    persistence.YamlPreloader._resolve_bare_default_args(test_obj)
    self.assertIsInstance(test_obj["a"].arg1.arg1, DummyClass)
    self.assertIsInstance(test_obj["b"].arg1, DummyClass2)
    self.assertIsInstance(test_obj["b"].arg1.arg1, DummyClass)

  def test_resolve_bare_default_args_illegal(self):
    test_obj = yaml.load("""
                         a: !DummyClassForgotBare {}
                         """)
    with self.assertRaises(ValueError):
      persistence.YamlPreloader._resolve_bare_default_args(test_obj)

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
    persistence.YamlPreloader._format_strings(test_obj, {"V1":"val1", "V2":"val2"})
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

class DummyArgClass(persistence.Serializable):
  yaml_tag = "!DummyArgClass"
  @persistence.serializable_init
  def __init__(self, arg1, arg2):
    pass # arg1 and arg2 are purposefully not kept
class DummyArgClass2(persistence.Serializable):
  yaml_tag = "!DummyArgClass2"
  @persistence.serializable_init
  def __init__(self, v):
    self.v = v

class TestSaving(unittest.TestCase):
  def setUp(self):
    events.clear()
    xnmt.resolved_serialize_params = {}
    yaml.add_representer(DummyArgClass, xnmt.init_representer)
    yaml.add_representer(DummyArgClass2, xnmt.init_representer)
    self.out_dir = os.path.join("test", "tmp")
    utils.make_parent_dir(os.path.join(self.out_dir, "asdf"))
    self.model_file = os.path.join(self.out_dir, "saved.mod")
    param_collections.ParamManager.init_param_col()
    param_collections.ParamManager.param_col.model_file = self.model_file

  def test_shallow(self):
    test_obj = yaml.load("""
                         a: !DummyArgClass
                           arg1: !DummyArgClass2
                             _xnmt_id: id1
                             v: some_val
                           arg2: !Ref { name: id1 }
                         """)
    preloaded = persistence.YamlPreloader.preload_obj(root=test_obj,exp_name="exp1",exp_dir=self.out_dir)
    initalized = persistence.initialize_if_needed(preloaded)
    persistence.save_to_file(self.model_file, initalized)

  def test_mid(self):
    test_obj = yaml.load("""
                         a: !DummyArgClass
                           arg1: !DummyArgClass2
                             v: !DummyArgClass2
                               _xnmt_id: id1
                               v: some_val
                           arg2: !DummyArgClass2
                             v: !Ref { name: id1 }
                         """)
    preloaded = persistence.YamlPreloader.preload_obj(root=test_obj,exp_name="exp1",exp_dir=self.out_dir)
    initalized = persistence.initialize_if_needed(preloaded)
    persistence.save_to_file(self.model_file, initalized)

  def test_deep(self):
    test_obj = yaml.load("""
                         a: !DummyArgClass
                           arg1: !DummyArgClass2
                             v: !DummyArgClass2
                               v: !DummyArgClass2
                                 _xnmt_id: id1
                                 v: some_val
                           arg2: !DummyArgClass2
                             v: !DummyArgClass2
                               v: !Ref { name: id1 }
                         """)
    preloaded = persistence.YamlPreloader.preload_obj(root=test_obj,exp_name="exp1",exp_dir=self.out_dir)
    initalized = persistence.initialize_if_needed(preloaded)
    persistence.save_to_file(self.model_file, initalized)

  @unittest.expectedFailure # TODO: need to fix resolving references inside lists
  def test_double_ref(self):
    test_obj = yaml.load("""
                         a: !DummyArgClass
                           arg1: !DummyArgClass2
                             _xnmt_id: id1
                             v: some_val
                           arg2:
                             - !Ref { name: id1 }
                             - !Ref { name: id1 }
                         """)
    preloaded = persistence.YamlPreloader.preload_obj(root=test_obj,exp_name="exp1",exp_dir=self.out_dir)
    initalized = persistence.initialize_if_needed(preloaded)
    persistence.save_to_file(self.model_file, initalized)



  def tearDown(self):
    try:
      if os.path.isdir(os.path.join("test","tmp")):
        shutil.rmtree(os.path.join("test","tmp"))
    except:
      pass

class TestReferences(unittest.TestCase):
  def setUp(self):
    events.clear()
    xnmt.resolved_serialize_params = {}
    yaml.add_representer(DummyArgClass, xnmt.init_representer)
    yaml.add_representer(DummyArgClass2, xnmt.init_representer)
    self.out_dir = os.path.join("test", "tmp")
    utils.make_parent_dir(os.path.join(self.out_dir, "asdf"))
    self.model_file = os.path.join(self.out_dir, "saved.mod")
    param_collections.ParamManager.init_param_col()
    param_collections.ParamManager.param_col.model_file = self.model_file

  def test_simple_reference(self):
    test_obj = yaml.load("""
                         !DummyArgClass
                         arg1: !DummyArgClass
                           arg1: !DummyArgClass2 { v: some_val }
                           arg2: !DummyArgClass2 { v: some_other_val }
                         arg2: !Ref { path: arg1 }
                         """)
    preloaded = persistence.YamlPreloader.preload_obj(root=test_obj,exp_name="exp1",exp_dir=self.out_dir)
    initialized = persistence.initialize_if_needed(preloaded)
    dump = persistence._dump(initialized)
    reloaded = yaml.load(dump)
    if isinstance(reloaded.arg1, persistence.Ref):
      reloaded.arg1, reloaded.arg2 = reloaded.arg2, reloaded.arg1
    self.assertIsInstance(reloaded.arg1, DummyArgClass)
    self.assertIsInstance(reloaded.arg2, persistence.Ref)
    self.assertIsInstance(reloaded.arg1.arg1, DummyArgClass2)
    self.assertIsInstance(reloaded.arg1.arg2, DummyArgClass2)

if __name__ == '__main__':
  unittest.main()
