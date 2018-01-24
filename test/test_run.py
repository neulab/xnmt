import unittest

import os, shutil, sys

if not any(a.startswith("--settings") for a in sys.argv): sys.argv.insert(1, "--settings=settings.unittest")

import xnmt.xnmt_run_experiments as run
import xnmt.events

class TestRunningConfig(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()

  def test_component_sharing(self):
    run.main(["test/config/component_sharing.yaml"])

  def test_encoders(self):
    run.main(["test/config/encoders.yaml"])

  def test_forced(self):
    run.main(["test/config/forced.yaml"])

  def test_load_model(self):
    run.main(["test/config/load_model.yaml"])

  def test_multi_task(self):
    run.main(["test/config/multi_task.yaml"])

  def test_preproc(self):
    run.main(["test/config/preproc.yaml"])

  def test_prior_segmenting(self):
    run.main(["test/config/prior_segmenting.yaml"])

  def test_random_search_test_params(self):
    run.main(["test/config/random_search_test_params.yaml"])

  def test_random_search_train_params(self):
    run.main(["test/config/random_search_train_params.yaml"])

  def test_reload(self):
    run.main(["test/config/reload.yaml"])

  def test_reload_exception(self):
    with self.assertRaises(ValueError) as context:
      run.main(["test/config/reload_exception.yaml"])
    self.assertEqual(str(context.exception), 'VanillaLSTMGates: x_t has inconsistent dimension')

  def test_report(self):
    run.main(["test/config/report.yaml"])

  @unittest.expectedFailure # TODO: these tests need to be fixed
  def test_retrieval(self):
    run.main(["test/config/retrieval.yaml"])

  def test_segmenting(self):
    run.main(["test/config/segmenting.yaml"])

  def test_speech(self):
    run.main(["test/config/speech.yaml"])

  def test_standard(self):
    run.main(["test/config/standard.yaml"])

  def test_transformer(self):
    run.main(["test/config/transformer.yaml"])
    
  def test_translator_loss(self):
    run.main(["test/config/translator_loss.yaml"])

  def tearDown(self):
    if os.path.isdir("test/tmp"):
      shutil.rmtree("test/tmp")

if __name__ == "__main__":
  unittest.main()
