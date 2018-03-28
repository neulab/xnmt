# all Serializable objects must be imported here in order to be parsable
# using the !Classname YAML syntax
import xnmt.attender
import xnmt.batcher
import xnmt.conv
import xnmt.decoder
import xnmt.embedder
import xnmt.eval_task
import xnmt.evaluator
import xnmt.experiment
import xnmt.ff
import xnmt.hyper_parameters
import xnmt.inference
import xnmt.input
import xnmt.input_reader
import xnmt.lstm
import xnmt.exp_global
import xnmt.optimizer
import xnmt.param_init
import xnmt.preproc_runner
import xnmt.pyramidal
import xnmt.residual
import xnmt.retriever
import xnmt.segmenting_composer
import xnmt.segmenting_encoder
import xnmt.serialize.tree_tools
import xnmt.specialized_encoders
import xnmt.training_regimen
import xnmt.training_task
import xnmt.transformer
import xnmt.translator
import xnmt.bow_predictor

def init_representer(dumper, obj):
  if not hasattr(obj, "resolved_serialize_params") and not hasattr(obj, "serialize_params"):
    raise RuntimeError(f"Serializing object {obj} that does not possess serialize_params, probably because it was created programmatically, is not possible.")
  if hasattr(obj, "resolved_serialize_params"):
    serialize_params = obj.resolved_serialize_params
  else:
    serialize_params = obj.serialize_params
  return dumper.represent_mapping('!' + obj.__class__.__name__, serialize_params)

from xnmt.serialize.serializable import Serializable
import yaml

for SerializableChild in Serializable.__subclasses__():
  yaml.add_representer(SerializableChild, init_representer)
