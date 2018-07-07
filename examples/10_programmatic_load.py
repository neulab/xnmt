# This demonstrates how to load the model trained using ``09_programmatic.py``
# the programmatic way and for the purpose of evaluating the model.

import os

import xnmt.tee
from xnmt.param_collection import ParamManager
from xnmt.persistence import initialize_if_needed, YamlPreloader, LoadSerialized, save_to_file

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

uninitialized_experiment = YamlPreloader.preload_obj(load_experiment, exp_dir=EXP_DIR, exp_name=EXP)
loaded_experiment = initialize_if_needed(uninitialized_experiment)

# if we were to continue training, we would need to set a save model file like this:
# ParamManager.param_col.model_file = model_file
ParamManager.populate()

# run experiment
loaded_experiment(save_fct=lambda: save_to_file(model_file, loaded_experiment))
