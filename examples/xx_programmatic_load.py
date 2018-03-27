import os

from xnmt.experiment import Experiment
from xnmt.serialize.serializer import YamlSerializer
from xnmt.serialize.options import OptionParser

EXP_DIR = os.path.dirname(__file__)
EXP = "programmatic-load"

model_file = f"{EXP_DIR}/models/{EXP}.mod"

load_experiment = Experiment(
  load = f"{EXP_DIR}/models/programmatic.mod",
  overwrite = [
    {"path" : "train", "val" : None}
  ]
)

config_parser = OptionParser()
uninitialized_experiment = config_parser.parse_loaded_experiment(load_experiment, exp_dir=EXP_DIR, exp_name=EXP)
loaded_experiment = YamlSerializer().initialize_if_needed(uninitialized_experiment)
ParamManager.populate()
exp_global = loaded_experiment.exp_global

# run experiment
loaded_experiment(save_fct=lambda: YamlSerializer().save_to_file(model_file,
                                                                 loaded_experiment,
                                                                 exp_global.dynet_param_collection))

