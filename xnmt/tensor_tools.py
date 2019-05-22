"""
A collection of backend-agnostic utilities.

The goal of this module is to provide a bridge between DyNet and Torch code, by providing commonly used functionality
that allows writing code that works with either backend.

This is *not* meant as a complete wrapper around each backend. Rather, only important high level functionality is
covered, dealing with tensor dimensions, reshaping, aggregation, etc.

"""

from abc import ABC
from typing import Callable, Sequence
import numbers

import xnmt
from xnmt.settings import settings
from xnmt import param_collections, trace

class Tensor(ABC): pass

if xnmt.backend_dynet:
  import dynet as dy
  Tensor.register(dy.Expression)
if xnmt.backend_torch:
  import torch
  import torch.nn as nn
  Tensor.register(torch.Tensor)


def reset_graph(zero_grad: bool = True) -> None:
  """
  Reset graph and/or gradients.

  DyNet case: reset computation graph (this is done implicitly by Pytorch garbage collection)
  Pytorch case: zero gradients (unless zero_grad is set to False). This is done automatically upon update() in DyNet.

  Args:
    zero_grad: Whether to zero gradients with Pytorch backend.
  """
  if xnmt.backend_dynet:
    dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
  if xnmt.backend_torch:
    torch.autograd.set_detect_anomaly(settings.CHECK_VALIDITY)
    if zero_grad:
      param_collections.ParamManager.global_collection().zero_grad()
  trace.reset()

def sent_len(x):
  """
  Get the sentence length of a sequence tensor.

  Args:
    x: Tensor of matrix shape (+ batch dim)

  Returns:
    Sentence length.
  """
  if xnmt.backend_dynet:
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def sent_len_transp(x):
  """
  Get the sentence length of a transposed sequence tensor (with flipped hidden/time dims).

  Args:
    x: Tensor of matrix shape (+ batch dim)

  Returns:
    Sentence length.
  """
  if xnmt.backend_dynet:
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def batch_size(x: Tensor) -> int:
  """
  Get batch size of tensor.

  Args:
    x: a DyNet expression or PyTorch tensor.

  Returns:
    The batch size.
  """
  if xnmt.backend_dynet:
    return x.dim()[1]
  else:
    return x.size()[0]

def hidden_size(x: Tensor) -> int:
  """
  Get the hidden dimension of a batched tensor, e.g. a vector or a sequence tensor.

  Args:
    x: batched tensor

  Returns:
    vector size
  """
  if xnmt.backend_dynet:
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def hidden_size_transp(x: Tensor) -> int:
  """
  Get the vector dimension of a transposed batched vector.

  Args:
    x: vector

  Returns:
    vector size
  """
  if xnmt.backend_dynet:
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def dim_desc(x: Tensor) -> tuple:
  """
  Get a tuple describing the tensor dimensions.

  DyNet case: ((dim_1, ..dim_n), batch_size)
  PyTorch case: (batch_size, dim_n, .., dim_1)

  Args:
    x: tensor

  Returns:
    dimension description
  """
  if xnmt.backend_dynet:
    return x.dim()
  else:
    return x.size()

def merge_time_batch_dims(x: Tensor) -> Tensor:
  """
  Pack the time dimension into the batch dimension.

  Args:
    x: input tensor

  Returns:
    output tensor
  """
  if xnmt.backend_dynet:
    ((hidden_dim, seq_len), batch_size_) = x.dim()
    return dy.reshape(x, (hidden_dim,), batch_size=batch_size_ * seq_len)
  else:
    batch_size_, seq_len, hidden_dim = x.size()
    return x.view((batch_size_ * seq_len, hidden_dim))

def unmerge_time_batch_dims(x: Tensor, batch_size_: numbers.Integral) -> Tensor:
  """
  Undo packing of the time dimension into the batch dimension.

  Args:
    x: input tensor
    batch_size_: original batch size

  Returns:
    output tensor
  """
  if xnmt.backend_dynet:
    seq_len = x.dim()[1] // batch_size_
    hidden_dim = x.dim()[0]
    if hidden_dim == (1,): hidden_dim = tuple()
    return dy.reshape(x, hidden_dim + (seq_len,), batch_size=batch_size_)
  else:
    seq_len = x.size()[0] // batch_size_
    hidden_dim = x.size()[1:]
    return x.view((batch_size_, seq_len) + hidden_dim)

def aggregate_masked_loss(x: Tensor, mask: xnmt.batchers.Mask=None) -> Tensor:
  """
  Aggregate loss values for unmasked entries.

  Args:
    x: Batched sequence of losses.
    mask: An optional mask for the case of outputs of unequal lengths.

  Returns:
    Batched sequence of losses, with masked ones zeroed out.
  """
  if xnmt.backend_dynet:
    if mask:
      x = dy.cmult(x, dy.inputTensor(1.0 - mask.np_arr.T, batched=True))
    return dy.sum_elems(x)
  else:
    if mask:
      x = torch.mul(x, torch.as_tensor(1.0 - mask.np_arr, dtype=x.dtype, device=xnmt.device))
    return torch.sum(x, dim=tuple(range(1, len(x.size())))) # sum over all but batch elems

def esum(x: Sequence[Tensor]) -> Tensor:
  """
  Perform an elementwise sum over all the given expressions.

  Args:
    x: list of tensor expressions of equal size to sum over.

  Returns:
    Summed tensor.
  """
  if xnmt.backend_dynet:
    return dy.esum(x)
  else:
    return sum(x)

def zeroes(hidden_dim: numbers.Integral, batch_size: numbers.Integral=1) -> Tensor:
  """
  Create a possibly batched zero vector.

  Args:
    hidden_dim: vector size
    batch_size: batch size

  Returns:
    DyNet expression of size ((hidden_dim,),batch_size) or PyTorch tensor of size (batch_size,hidden_dim)
  """
  if xnmt.backend_dynet:
    return dy.zeroes((hidden_dim,), batch_size=batch_size)
  else:
    return torch.zeros(size=(batch_size, hidden_dim,), device=xnmt.device)

def concatenate(l: Sequence[Tensor]) -> Tensor:
  """
  Stack batched vectors to form a longer batched vector.

  Args:
    l: list of batched vectors (DyNet dims: ((vec_size),batch_size); PyTorch dims: (batch_size,vec_size)).

  Returns:
    A batched vector.
  """
  if xnmt.backend_dynet:
    return dy.concatenate(l)
  else:
    return torch.cat(l, dim=1)

def npvalue(t: Tensor) -> 'np.ndarray':
  """
  Numpy array in column-major format (i.e., results will be in DyNet format, regardless of backend)

  Args:
    t: Tensor

  Returns:
    Numpy array
  """
  if xnmt.backend_dynet:
    return t.npvalue()
  else:
    ret = t.cpu().data.numpy()
    if batch_size(t)==1 and t.dim()>1:
      ret = ret.squeeze(0)
    return ret.T

def average(l: Sequence[Tensor]) -> Tensor:
  """
  Perform an elementwise average over all the given tensor expressions.

  Args:
    l: list of tensor expressions of matching size.

  Returns:
    Averaged tensor expression.
  """
  if xnmt.backend_dynet:
    return dy.average(l)
  else:
    return sum(l) / len(l)

def dropout(t: Tensor, p: numbers.Real) -> Tensor:
  """
  Dropout elements of the given tensor with probability p, and rescale accordingly.

  Args:
    t: input tensor
    p: dropout probability

  Returns:
    output tensor
  """
  if xnmt.backend_dynet:
    return dy.dropout(t, p)
  else:
    return nn.Dropout(p=p)(t)


def identity(x: Tensor) -> Tensor:
  """
  Identity function.

  Args:
    x: input

  Returns:
    output, same as input.
  """
  return x

def activation_by_name(activation: str) -> Callable[[Tensor],Tensor]:
  """
  Get a callable activation function, resolving potential different namings between backends.

  Args:
    activation: name of activation (tanh|rectify|relu|sigmoid|elu|selu|asinh|identity)

  Returns:
    A unary tensor activation function.
  """
  if activation == 'tanh':
    return dy.tanh if xnmt.backend_dynet else torch.tanh
  elif activation in ['rectify','relu']:
    return dy.rectify if xnmt.backend_dynet else torch.relu
  elif activation == 'sigmoid':
    return dy.sigmoid if xnmt.backend_dynet else torch.sigmoid
  elif activation == 'elu':
    return dy.elu if xnmt.backend_dynet else torch.elu
  elif activation == 'selu':
    return dy.selu if xnmt.backend_dynet else torch.selu
  elif activation == 'asinh':
    if xnmt.backend_dynet:
      return dy.asinh
    else:
      raise ValueError(f"Unknown activation {activation}")
  elif activation == 'identity':
    return identity
  else:
    raise ValueError(f"Unknown activation {activation}")
