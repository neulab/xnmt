import dynet as dy
import torch
from abc import ABC

class Tensor(ABC): pass

Tensor.register(dy.Expression)
Tensor.register(torch.Tensor)

def sent_len(x):
  if isinstance(x, dy.Expression):
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def sent_len_transp(x):
  if isinstance(x, dy.Expression):
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def batch_size(x):
  if isinstance(x, dy.Expression):
    return x.dim()[1]
  else:
    return x.size()[0]

def hidden_size(x):
  if isinstance(x, dy.Expression):
    return x.dim()[0][0]
  else:
    return x.size()[-1]

def hidden_size_transp(x):
  if isinstance(x, dy.Expression):
    return x.dim()[0][-1]
  else:
    return x.size()[1]

def dim_desc(x):
  if isinstance(x, dy.Expression):
    return x.dim()
  else:
    return x.size()
