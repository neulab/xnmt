from typing import List

import dynet as dy

from xnmt.transducers import base
from xnmt.modelparts import transforms as modelparts_transforms
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt import events, expression_seqs, norms, param_collections, param_initializers

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

class NinTransducer(base.SeqTransducer, Serializable):
  """
  Network-in-network transducer following Lin et al. (2013): Network in Network; https://arxiv.org/pdf/1312.4400.pdf

  Here, this is a shared linear transformation across time steps, followed by batch normalization and a non-linearity.

  Args:
    input_dim: dimension of inputs
    hidden_dim: dimension of outputs
    use_proj: whether to enable the linear projection
    use_bn: whether to enable batch norm
    batch_norm: automatically set
    nonlinearity: name of a unary DyNet operation (or ``id`` to use the identity transformation)
    downsampling_factor: if > 1, feed adjacent time steps to the linear projections to downsample the sequence to a
                         shorter length
    param_init: how to initialize the projection matrix
  """

  yaml_tag = "!NinTransducer"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               use_proj: bool = True,
               use_bn: bool = True,
               batch_norm=None,
               nonlinearity: str = "rectify",
               downsampling_factor: int = 1,
               param_init = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))) -> None:
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.use_proj = use_proj
    self.use_bn = use_bn
    if use_bn or use_proj:
      param_col = param_collections.ParamManager.my_params(self)
    if nonlinearity == "id":
      self.nonlinearity = lambda x: x
    else:
      self.nonlinearity = getattr(dy, nonlinearity)
    self.downsampling_factor = downsampling_factor
    if downsampling_factor < 1: raise ValueError("downsampling_factor must be >= 1")
    if use_proj:
      dim_proj = (hidden_dim, input_dim * downsampling_factor)
      self.p_proj = param_col.add_parameters(dim=dim_proj, init=param_init.initializer(dim_proj))
    elif hidden_dim != input_dim * downsampling_factor:
      raise ValueError("disabling projections requires hidden_dim == input_dim*downsampling_factor")
    if self.use_bn:
      self.batch_norm = self.add_serializable_component("batch_norm", batch_norm,
                                                        lambda: norms.BatchNorm(hidden_dim, 2, time_first=False))

  def transduce(self, es: expression_seqs.ExpressionSequence):
    """
    Args:
      es: expression sequence of dimensions input_dim x time
    Return:
      expression sequence;
              if use_proj: dimensions = hidden x ceil(time/downsampling_factor)
              else:        dimensions = (input_dim*downsampling_factor) x ceil(time/downsampling_factor)
    """
    if not es.dim()[0][0] == self.input_dim:
      raise ValueError(f"This NiN Layer requires inputs of hidden dim {self.input_dim}, got {es.dim()[0][0]}.")

    if self.use_proj:
      if len(es) % self.downsampling_factor != 0:
        raise ValueError(
          "For downsampling with activated use_proj, sequence lengths must be multiples of the total reduce factor. "
          "Configure batcher accordingly.")
    if es.mask is None:
      mask_out = None
    else:
      if self.downsampling_factor == 1:
        mask_out = es.mask
      else:
        mask_out = es.mask.lin_subsampled(self.downsampling_factor)

    expr_tensor = es.as_tensor()
    batch_size = expr_tensor.dim()[1]
    seq_len = expr_tensor.dim()[0][-1]
    if self.use_proj:
      reshaped = dy.reshape(expr_tensor, (self.input_dim * self.downsampling_factor,),
                            batch_size=seq_len / self.downsampling_factor * batch_size)
      projected = dy.parameter(self.p_proj) * reshaped
      projected = dy.reshape(projected, (self.hidden_dim, seq_len / self.downsampling_factor), batch_size=batch_size)
    else:
      projected = dy.reshape(expr_tensor, (self.input_dim * self.downsampling_factor, seq_len), batch_size=batch_size)

    if self.use_bn:
      bn_layer = self.batch_norm.transform(projected, train=self.train, mask=mask_out)
      nonlin = self.nonlinearity(bn_layer)
    else:
      nonlin = self.nonlinearity(projected)

    self._final_states = [base.FinalTransducerState(nonlin[-1])]

    return expression_seqs.ExpressionSequence(expr_tensor=nonlin, mask=mask_out)

  def get_final_states(self) -> List[base.FinalTransducerState]:
    return self._final_states

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, *args, **kwargs):
    self._final_states = None
