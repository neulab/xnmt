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
import xnmt.classifier
import xnmt.conv
import xnmt.decoder
import xnmt.embedder
import xnmt.eval_task
import xnmt.evaluator
import xnmt.exp_global
import xnmt.experiment
import xnmt.fixed_size_att
import xnmt.hyper_parameters
import xnmt.inference
import xnmt.input
import xnmt.input_reader
import xnmt.lm
import xnmt.lstm
import xnmt.model_base
import xnmt.optimizer
import xnmt.param_init
import xnmt.positional
import xnmt.preproc_runner
import xnmt.pyramidal
import xnmt.reports
import xnmt.residual
import xnmt.retriever
import xnmt.scorer
import xnmt.self_attention
import xnmt.seq_labeler
import xnmt.specialized_encoders.tilburg_harwath
import xnmt.specialized_encoders.self_attentional_am
import xnmt.specialized_encoders.segmenting_encoder
import xnmt.training_regimen
import xnmt.training_task
import xnmt.transformer
import xnmt.translator
import xnmt.persistence
import xnmt.rl
import xnmt.compound_expr

resolved_serialize_params = {}

def init_representer(dumper, obj):
  if id(obj) not in resolved_serialize_params:
  # if len(resolved_serialize_params)==0:
    serialize_params = obj.serialize_params
  else:
    serialize_params = resolved_serialize_params[id(obj)]
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
  assert getattr(SerializableChild.__init__, "uses_serializable_init",
                 False), f"{SerializableChild.__name__}.__init__() must be wrapped in @serializable_init."
  seen_yaml_tags.add(SerializableChild.yaml_tag)
  yaml.add_representer(SerializableChild, init_representer)
