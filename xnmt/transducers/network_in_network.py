import numbers
from typing import List, Optional

from xnmt.transducers import base
from xnmt.modelparts import transforms as modelparts_transforms
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt import norms, param_initializers

class NinSeqTransducer(base.ModularSeqTransducer, Serializable):
  """
  Network-in-network transducer following Lin et al. (2013): Network in Network; https://arxiv.org/pdf/1312.4400.pdf

  Here, this is a shared linear transformation across time steps, followed by batch normalization and a non-linearity.

  Args:
    input_dim: dimension of inputs
    hidden_dim: dimension of outputs
    downsample_by: if > 1, feed adjacent time steps to the linear projections to downsample the sequence
    param_init: how to initialize the projection matrix
    projection: automatically set
    batch_norm: automatically set
    nonlinearity: automatically set
  """
  yaml_tag = "!NinSeqTransducer"

  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               downsample_by: numbers.Integral = 1,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               projection: Optional[base.TransformSeqTransducer] = None,
               batch_norm: Optional[norms.BatchNorm] = None,
               nonlinearity: Optional[base.TransformSeqTransducer] = None) -> None:
    self.projection = self.add_serializable_component("projection", projection,
                                                      lambda: base.TransformSeqTransducer(
                                                        modelparts_transforms.Linear(input_dim=input_dim*downsample_by,
                                                                                     output_dim=hidden_dim,
                                                                                     bias=False,
                                                                                     param_init=param_init),
                                                        downsample_by=downsample_by))
    self.batch_norm = self.add_serializable_component("batch_norm", batch_norm,
                                                      lambda: norms.BatchNorm(hidden_dim=hidden_dim,
                                                                              num_dim=2))
    self.nonlinearity = self.add_serializable_component("nonlinearity", nonlinearity,
                                                        lambda: base.TransformSeqTransducer(
                                                          modelparts_transforms.Cwise("rectify")
                                                        ))
    self.modules = [self.projection, self.batch_norm, self.nonlinearity]

  def get_final_states(self) -> List[base.FinalTransducerState]:
    return self.modules[-1].get_final_states()
