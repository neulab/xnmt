import unittest
import os, shutil

from xnmt.utils import has_cython
import xnmt.xnmt_run_experiments as run
import xnmt.events
import xnmt.tee

class TestRunningConfig(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()

  def tearDown(self):
    try:
      if os.path.isdir("test/tmp"):
        shutil.rmtree("test/tmp")
    except:
      pass

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_assemble(self):
    run.main(["test/config/assemble.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_autobatch_fail(self):
    with self.assertRaises(ValueError) as context:
      run.main(["test/config/autobatch-fail.yaml"])
    self.assertEqual(str(context.exception), 'AutobatchTrainingRegimen forces the batcher to have batch_size 1. Use update_every to set the actual batch size in this regimen.')

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_autobatch(self):
    run.main(["test/config/autobatch.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_cascade(self):
    run.main(["test/config/cascade.yaml"])

  def test_classifier(self):
    run.main(["test/config/classifier.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_component_sharing(self):
    run.main(["test/config/component_sharing.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_encoders(self):
    run.main(["test/config/encoders.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_ensembling(self):
    run.main(["test/config/ensembling.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_forced(self):
    run.main(["test/config/forced.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_lattice(self):
    run.main(["test/config/lattice.yaml"])

  def test_lm(self):
    run.main(["test/config/lm.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_load_model(self):
    run.main(["test/config/load_model.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_multi_task(self):
    run.main(["test/config/multi_task.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_multi_task_speech(self):
    run.main(["test/config/multi_task_speech.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_preproc(self):
    run.main(["test/config/preproc.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_pretrained_emb(self):
    run.main(["test/config/pretrained_embeddings.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_random_search_test_params(self):
    run.main(["test/config/random_search_test_params.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_random_search_train_params(self):
    run.main(["test/config/random_search_train_params.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_reload(self):
    run.main(["test/config/reload.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_segmenting(self):
    run.main(["test/config/seg_report.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_reload_exception(self):
    with self.assertRaises(ValueError) as context:
      run.main(["test/config/reload_exception.yaml"])
    self.assertEqual(str(context.exception), 'VanillaLSTMGates: x_t has inconsistent dimension 20, expecting 40')

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_report(self):
    run.main(["test/config/report.yaml"])

  @unittest.expectedFailure # TODO: these tests need to be fixed
  def test_retrieval(self):
    run.main(["test/config/retrieval.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_score(self):
    run.main(["test/config/score.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_self_attentional_am(self):
    run.main(["test/config/self_attentional_am.yaml"])

  def test_seq_labeler(self):
    run.main(["test/config/seq_labeler.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_speech(self):
    run.main(["test/config/speech.yaml"])

  @unittest.expectedFailure # TODO: these tests need to be fixed
  def test_speech_retrieval(self):
    run.main(["test/config/speech_retrieval.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  def test_standard(self):
    run.main(["test/config/standard.yaml"])

  @unittest.expectedFailure # TODO: these tests need to be fixed
  def test_transformer(self):
    run.main(["test/config/transformer.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_search_strategy_reinforce(self):
    run.main(["test/config/reinforce.yaml"])

  @unittest.skipUnless(xnmt.backend_dynet, "requires DyNet backend")
  @unittest.skipUnless(has_cython(), "requires cython to run")
  def test_search_strategy_minrisk(self):
    run.main(["test/config/minrisk.yaml"])



if __name__ == "__main__":
  unittest.main()
