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
