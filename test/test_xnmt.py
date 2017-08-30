import unittest

import test.test_batcher
import test.test_run

def run_from_module(module, verbosity=2):
  suite = unittest.TestLoader().loadTestsFromModule(module)
  unittest.TextTestRunner(verbosity=verbosity).run(suite)

# Run All the tests
run_from_module(test.test_batcher)
run_from_module(test.test_run)

