import os
os.environ['XNMT_SETTINGS'] = 'unittest'
import dynet_config

dynet_config.set(random_seed=2)