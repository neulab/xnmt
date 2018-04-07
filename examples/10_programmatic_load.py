# This demonstrates how to load the model trained using ``09_programmatic.py``
# the programmatic way and for the purpose of evaluating the model.

import logging
logger = logging.getLogger('xnmt')
import os

import xnmt.tee
from xnmt.param_collection import ParamManager
from xnmt.persistence import YamlSerializer, YamlPreloader, LoadSerialized

EXP_DIR = os.path.dirname(__file__)
EXP = "programmatic-load"

model_file = f"{EXP_DIR}/models/{EXP}.mod"
log_file = f"{EXP_DIR}/logs/{EXP}.log"

xnmt.tee.set_out_file(log_file)

ParamManager.init_param_col()

load_experiment = LoadSerialized(
  filename=f"{EXP_DIR}/models/programmatic.mod",
  overwrite=[
    {"path" : "train", "val" : None}
  ]
)
# TODO: prints the below, should debug:
# > populated a subset of DyNet weights from given data files: ['SimpleWordEmbedder.77f45fe2', 'UniLSTMSeqTransducer.aff22365', 'UniLSTMSeqTransducer.99318f39', 'MlpAttender.34f8a738', 'SimpleWordEmbedder.bdbb3a11', 'UniLSTMSeqTransducer.747cc597', 'MLP.ed561ae9', 'MlpSoftmaxDecoder.d7b1d2a3'].
#   Did not populate {'Linear.f3cfe057', 'Linear.3d2abbbd'}

uninitialized_experiment = YamlPreloader.preload_obj(load_experiment, exp_dir=EXP_DIR, exp_name=EXP)
loaded_experiment = YamlSerializer().initialize_if_needed(uninitialized_experiment)

# if we were to continue training, we would need to set a save model file like this:
# ParamManager.param_col.model_file = model_file
ParamManager.populate()
exp_global = loaded_experiment.exp_global

# run experiment
loaded_experiment(save_fct=lambda: YamlSerializer().save_to_file(model_file,
                                                                 loaded_experiment,
                                                                 exp_global.dynet_param_collection))

