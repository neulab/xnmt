from xnmt.transducers import base
from xnmt.modelparts import transforms as modelparts_transforms
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt import norms, param_initializers

class NinLayer(base.ModularSeqTransducer, Serializable):
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
  yaml_tag = "!NinLayer"

  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               downsample_by: int = 1,
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               projection=None, batch_norm=None, nonlinearity=None):
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

  def get_final_states(self):
    return self.modules[-1].get_final_states()
