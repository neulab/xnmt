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
file_logger = logging.getLogger('xnmt_file')

import _dynet
dyparams = _dynet.DynetParams()
dyparams.from_args()


# all Serializable objects must be imported here in order to be parsable
# using the !Classname YAML syntax
import xnmt.batchers
import xnmt.eval.metrics
import xnmt.eval.tasks
import xnmt.experiments
import xnmt.hyper_params
import xnmt.inferences
import xnmt.input_readers
import xnmt.modelparts.attenders
import xnmt.modelparts.bridges
import xnmt.modelparts.decoders
import xnmt.modelparts.embedders
import xnmt.modelparts.scorers
import xnmt.modelparts.transforms
import xnmt.models.base
import xnmt.models.classifiers
import xnmt.models.language_models
import xnmt.models.retrievers
import xnmt.models.sequence_labelers
import xnmt.models.translators
import xnmt.optimizers
import xnmt.param_initializers
import xnmt.persistence
import xnmt.reports
import xnmt.rl
import xnmt.specialized_encoders.segmenting_encoder
import xnmt.specialized_encoders.self_attentional_am
import xnmt.specialized_encoders.tilburg_harwath
import xnmt.train.regimens
import xnmt.train.tasks
import xnmt.transducers.convolution
import xnmt.transducers.lattice
import xnmt.transducers.network_in_network
import xnmt.transducers.positional
import xnmt.transducers.pyramidal
import xnmt.transducers.recurrent
import xnmt.transducers.residual
import xnmt.transducers.self_attention
import xnmt.transformer


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
