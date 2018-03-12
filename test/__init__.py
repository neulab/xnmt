import sys

if not any(a.startswith("--settings") for a in sys.argv): sys.argv.insert(1, "--settings=settings.unittest")

import dynet_config

dynet_config.set(random_seed=2)
