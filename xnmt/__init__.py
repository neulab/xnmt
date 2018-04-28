import os
import sys

# No support for python2
if sys.version_info[0] == 2:
  raise RuntimeError("XNMT does not support python2 any longer.")

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
  sys.path.append(package_dir)

import logging
logger = logging.getLogger('xnmt')
yaml_logger = logging.getLogger('yaml')

import _dynet
dyparams = _dynet.DynetParams()
dyparams.from_args()


# all Serializable objects must be imported here in order to be parsable
# using the !Classname YAML syntax
import xnmt.attender
import xnmt.batcher
import xnmt.conv
import xnmt.decoder
import xnmt.embedder
import xnmt.eval_task
import xnmt.evaluator
import xnmt.exp_global
import xnmt.experiment
import xnmt.ff
import xnmt.hyper_parameters
import xnmt.inference
import xnmt.input
import xnmt.input_reader
import xnmt.lstm
import xnmt.mlp
import xnmt.exp_global
import xnmt.optimizer
import xnmt.param_init
import xnmt.preproc_runner
import xnmt.pyramidal
import xnmt.residual
import xnmt.retriever
import xnmt.segmenting_composer
import xnmt.segmenting_encoder
import xnmt.specialized_encoders
import xnmt.training_regimen
import xnmt.training_task
import xnmt.transformer
import xnmt.translator
import xnmt.persistence

def init_representer(dumper, obj):
  if not hasattr(obj, "resolved_serialize_params") and not hasattr(obj, "serialize_params"):
    raise RuntimeError(f"Serializing object {obj} that does not possess serialize_params, probably because it was created programmatically, is not possible.")
  if hasattr(obj, "resolved_serialize_params"):
    serialize_params = obj.resolved_serialize_params
  else:
    serialize_params = obj.serialize_params
  return dumper.represent_mapping('!' + obj.__class__.__name__, serialize_params)

import yaml
seen_yaml_tags = set()
for SerializableChild in xnmt.persistence.Serializable.__subclasses__():
  assert hasattr(SerializableChild,
                 "yaml_tag") and SerializableChild.yaml_tag == f"!{SerializableChild.__name__}",\
    f"missing or misnamed yaml_tag attribute for class {SerializableChild.__name__}"
  assert SerializableChild.yaml_tag not in seen_yaml_tags, \
    f"encountered naming conflict: more than one class with yaml_tag='{SerializableChild.yaml_tag}'. " \
    f"Change to a unique class name."
  seen_yaml_tags.add(SerializableChild.yaml_tag)
  yaml.add_representer(SerializableChild, init_representer)
