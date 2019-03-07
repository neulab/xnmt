"""
A collection of backend-agnostic utilities.

The goal of this module is to provide a bridge between DyNet and Torch code, by providing commonly used functionality
that allows writing code that works with either backend.

This is *not* meant as a complete wrapper around each backend. Rather, only important high level functionality is
covered, dealing with tensor dimensions, reshaping, aggregation, etc.

"""

from abc import ABC

import xnmt
from xnmt.settings import settings

class Tensor(ABC): pass

if xnmt.backend_dynet:
  import dynet as dy
  Tensor.register(dy.Expression)
if xnmt.backend_torch:
  import torch
  Tensor.register(torch.Tensor)

def reset_graph():
  if xnmt.backend_dynet:
    dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)

def sent_len(x):
  if xnmt.backend_dynet:
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def sent_len_transp(x):
  if xnmt.backend_dynet:
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def batch_size(x):
  if xnmt.backend_dynet:
    return x.dim()[1]
  else:
    return x.size()[0]

def hidden_size(x):
  if xnmt.backend_dynet:
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def hidden_size_transp(x):
  if xnmt.backend_dynet:
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def dim_desc(x):
  if xnmt.backend_dynet:
    return x.dim()
  else:
    return x.size()

def merge_time_batch_dims(x):
  if xnmt.backend_dynet:
    ((hidden_dim, seq_len), batch_size_) = x.dim()
    return dy.reshape(x, (hidden_dim,), batch_size=batch_size_ * seq_len)
  else:
    batch_size_, seq_len, hidden_dim = x.size()
    return x.view((batch_size_ * seq_len, hidden_dim))

def unmerge_time_batch_dims(x, batch_size_):
  if xnmt.backend_dynet:
    seq_len = x.dim()[1] // batch_size_
    hidden_dim = x.dim()[0]
    if hidden_dim == (1,): hidden_dim = tuple()
    return dy.reshape(x, hidden_dim + (seq_len,), batch_size=batch_size_)
  else:
    seq_len = x.size()[0] // batch_size_
    hidden_dim = x.size()[2:]
    return x.view((batch_size_, seq_len) + hidden_dim)

def aggregate_masked_loss(x, mask=None):
  if xnmt.backend_dynet:
    if mask:
      x = dy.cmult(x, dy.inputTensor(1.0 - mask.np_arr.T, batched=True))
    return dy.sum_elems(x)
  else:
    if mask:
      x = torch.mul(x, torch.as_tensor(1.0 - mask.np_arr, dtype=x.dtype, device=xnmt.device))
    return torch.sum(x, dim=tuple(range(1, len(x.size())))) # sum over all but batch elems

def esum(x):
  if xnmt.backend_dynet:
    return dy.esum(x)
  else:
    return sum(x)