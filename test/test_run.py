import unittest
import xnmt.xnmt_run_experiments as run

class TestRunningConfig(unittest.TestCase):

  def test_random_search_test_params(self):
    run.main(["test/config/random_search_test_params.yaml"])

  def test_random_search_train_params(self):
    run.main(["test/config/random_search_train_params.yaml"])

  def test_report(self):
    run.main(["test/config/report.yaml"])

# TODO: these tests need to be fixed
#  def test_retrieval(self):
#    run.main(["test/config/retrieval.yaml"])
#
#  def test_segmenting(self):
#    run.main(["test/config/segmenting.yaml"])
#
#  def test_segmenting2(self):
#    run.main(["test/config/segmenting2.yaml"])

  def test_translator_report(self):
    run.main(["test/config/translator_report.yaml"])

  def test_translator(self):
    run.main(["test/config/translator.yaml"])

if __name__ == "__main__":
  unittest.main()
