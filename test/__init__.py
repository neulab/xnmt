import os
os.environ['XNMT_SETTINGS'] = 'unittest'

if os.environ.get('XNMT_BACKEND', "dynet")== 'dynet':
  import dynet_config
  dynet_config.set(random_seed=2)