import os
import sys
import argparse

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

base_arg_parser = argparse.ArgumentParser()
base_arg_parser.add_argument('--backend', type=str, default="dynet")
base_arg_parser.add_argument('--gpu', action='store_true')
args = base_arg_parser.parse_known_args(sys.argv)[0]
if args.backend=="torch":
  backend_dynet, backend_torch = False, True
  import torch
  if args.gpu:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
elif args.backend=="dynet":
  backend_dynet, backend_torch = True, False
else:
  raise ValueError(f"unknown backend {args.backend}")

import yaml

def no_init(*wrong, **backend): raise ValueError("wrong backend")
def require_dynet(x):
  x.xnmt_backend = "dynet"
  if backend_torch:
    x.__init__ = no_init
    if hasattr(x, "yaml_tag"):
      delattr(x, "yaml_tag")
  return x
def require_torch(x):
  x.xnmt_backend = "torch"
  if backend_dynet:
    x.__init__ = no_init
    if hasattr(x, "yaml_tag"):
      delattr(x, "yaml_tag")
  return x
def resolve_backend(a, b):
  expected_backend = "dynet" if backend_dynet else "torch"
  resolved = a if a.xnmt_backend==expected_backend else b
  if hasattr(resolved, "yaml_tag"):
    new_name = resolved.__name__[:len(resolved.yaml_tag)-1]
    assert resolved.__name__.startswith(new_name)
    resolved.__name__ = new_name
    yaml.loader.Loader.yaml_constructors[resolved.yaml_tag] = resolved.from_yaml
  return resolved


# all Serializable objects must be imported here in order to be parsable
# using the !Classname YAML syntax
import xnmt.batchers
import xnmt.eval.metrics
import xnmt.eval.tasks
import xnmt.experiments
import xnmt.hyper_params
import xnmt.inferences
import xnmt.input_readers
import xnmt.graph
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
import xnmt.simultaneous.simult_translators
import xnmt.simultaneous.simult_search_strategies
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

seen_yaml_tags = set()
for SerializableChild in xnmt.persistence.Serializable.__subclasses__():
  needed_for_backend = (backend_dynet and getattr(SerializableChild, "xnmt_backend", "dynet") == "dynet") or \
                       (backend_torch and getattr(SerializableChild, "xnmt_backend", "torch") == "torch")
  assert hasattr(SerializableChild, "yaml_tag"),\
    f"missing yaml_tag attribute for class {SerializableChild.__name__}"
  if needed_for_backend:
    assert SerializableChild.yaml_tag == f"!{SerializableChild.__name__}", \
      f"misnamed yaml_tag attribute for class {SerializableChild.__name__}"
    assert SerializableChild.yaml_tag not in seen_yaml_tags, \
      f"encountered naming conflict: more than one class with yaml_tag='{SerializableChild.yaml_tag}'. " \
      f"Change to a unique class name."
    assert getattr(SerializableChild.__init__, "uses_serializable_init",
                   False), f"{SerializableChild.__name__}.__init__() must be wrapped in @serializable_init."
    seen_yaml_tags.add(SerializableChild.yaml_tag)
    yaml.add_representer(SerializableChild, init_representer)
