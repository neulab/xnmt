from typing import List, Optional, Sequence

import numpy as np

import xnmt
from xnmt import batchers
from xnmt import tensor_tools as tt

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch


class BaseExpressionSequence(object):
  """A class to represent a sequence of expressions.

  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self,
               expr_list: Optional[Sequence[tt.Tensor]] = None,
               expr_tensor: Optional[tt.Tensor] = None,
               expr_transposed_tensor: Optional[tt.Tensor] = None,
               mask: Optional['batchers.Mask'] = None,
               tensor_type = None) -> None:
    """Constructor.

    Args:
      expr_list: a python list of expressions
      expr_tensor: a tensor where last dimension are the sequence items
      expr_transposed_tensor: a tensor in transposed form (first dimension are sequence items)
      mask: an optional mask object indicating what positions in a batched tensor should be masked
      tensor_type: expected tensor type (dy.Expression or torch.Tensor)
    Raises:
      valueError: raises an exception if neither expr_list nor expr_tensor are given,
                  or if both have inconsistent length
    """
    self.expr_list = expr_list
    self.expr_tensor = expr_tensor
    self.expr_transposed_tensor = expr_transposed_tensor
    self.mask = mask
    if not (self.expr_list is not None or self.expr_tensor is not None or self.expr_transposed_tensor is not None):
      raise ValueError("must provide expr_list or expr_tensor")
    if self.expr_list is not None and self.expr_tensor is not None:
      if len(self.expr_list) != tt.sent_len(self.expr_tensor):
        raise ValueError("expr_list and expr_tensor must be of same length")
    if expr_list is not None:
      if not isinstance(expr_list,list):
        raise ValueError("expr_list must be list, was:", type(expr_list))
      if not isinstance(expr_list[0], tensor_type):
        raise ValueError(f"expr_list must contain {tensor_type}, found:", type(expr_list[0]))
      for e in expr_list[1:]:
        if tt.dim_desc(e) != tt.dim_desc(expr_list[0]):
          raise AssertionError()
    if expr_tensor is not None:
      if not isinstance(expr_tensor, tensor_type):
        raise ValueError(f"expr_tensor must be {tensor_type}, was:", type(expr_tensor))
    if expr_transposed_tensor is not None:
      if not isinstance(expr_transposed_tensor, tensor_type):
        raise ValueError(f"expr_transposed_tensor must be {tensor_type}, was:", type(expr_transposed_tensor))

  def __len__(self):
    """Return length.

    Returns:
      length of sequence
    """
    return self.sent_len()

  def sent_len(self):
    """Return length.

    Returns:
      length of sequence
    """
    if self.expr_list: return len(self.expr_list)
    elif self.expr_tensor is not None: return tt.sent_len(self.expr_tensor)
    else: return tt.sent_len_transp(self.expr_transposed_tensor)

  def batch_size(self):
    """Return length.

    Returns:
      length of sequence
    """
    if self.expr_list: return tt.batch_size(self.expr_list[0])
    elif self.expr_tensor is not None: return tt.batch_size(self.expr_tensor)
    else: return tt.batch_size(self.expr_transposed_tensor)

  def __iter__(self):
    """Return iterator.

    Returns:
      iterator over the sequence; results in explicit conversion to list
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return iter(self.expr_list)

  def __getitem__(self, key):
    """Get a single item.

    Returns:
      sequence item (expression); does not result in explicit conversion to list
    """
    raise NotImplementedError()

  def as_list(self) -> List[tt.Tensor]:
    """Get a list.

    Returns:
      the whole sequence as a list with each element one of the embeddings.
    """
    raise NotImplementedError()

  def has_list(self) -> bool:
    """
    Returns:
      False if as_list() will result in creating additional expressions, True otherwise
    """
    return self.expr_list is not None

  def as_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each column is one of the embeddings.
    """
    raise NotImplementedError()

  def has_tensor(self) -> bool:
    """
    Returns:
      False if as_tensor() will result in creating additional expressions, True otherwise
    """
    return self.expr_tensor is not None

  def as_transposed_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each row is one of the embeddings.
    """
    raise NotImplementedError()

  def has_transposed_tensor(self) -> bool:
    """
    Returns:
      False if as_transposed_tensor() will result in creating additional expressions, True otherwise
    """
    return self.expr_transposed_tensor is not None

  def dim(self) -> tuple:
    """
    Return dimension of the expression sequence

    Returns:
      result of self.as_tensor().size(), without explicitly constructing that tensor
    """
    if self.expr_list: return (self.expr_list[0].size()[0], len(self.expr_list)) + self.expr_list[0].size()[1:]
    elif self.expr_tensor is not None: return self.expr_tensor.size()
    elif self.expr_transposed_tensor is not None:
      return (self.expr_transposed_tensor.size()[1], self.expr_transposed_tensor.size()[0]) + self.expr_transposed_tensor.size()[2:]
    else:
      raise NotImplementedError()

@xnmt.require_dynet
class ExpressionSequenceDynet(BaseExpressionSequence):
  """A class to represent a sequence of expressions.

  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self,
               expr_list: Optional[Sequence[tt.Tensor]] = None,
               expr_tensor: Optional[tt.Tensor] = None,
               expr_transposed_tensor: Optional[tt.Tensor] = None,
               mask: Optional['batchers.Mask'] = None) -> None:
    """Constructor.

    Args:
      expr_list: a python list of expressions
      expr_tensor: a tensor where last dimension are the sequence items
      expr_transposed_tensor: a tensor in transposed form (first dimension are sequence items)
      mask: an optional mask object indicating what positions in a batched tensor should be masked
    Raises:
      valueError: raises an exception if neither expr_list nor expr_tensor are given,
                  or if both have inconsistent length
    """
    super().__init__(expr_list=expr_list, expr_tensor=expr_tensor, expr_transposed_tensor=expr_transposed_tensor,
                     mask=mask, tensor_type=dy.Expression)

  def __getitem__(self, key):
    """Get a single item.

    Returns:
      sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else:
      if key < 0: key += len(self)
      if self.expr_tensor:
        return dy.pick(self.expr_tensor, key, dim=len(self.expr_tensor.dim()[0])-1)
      else:
        return dy.pick(self.expr_transposed_tensor, key, dim=0)

  def as_list(self) -> List[tt.Tensor]:
    """Get a list.

    Returns:
      the whole sequence as a list with each element one of the embeddings.
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return self.expr_list

  def as_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each column is one of the embeddings.
    """
    if self.expr_tensor is None:
      if self.expr_list:
        self.expr_tensor = dy.concatenate_cols(self.expr_list)
      else:
        self.expr_tensor = dy.transpose(self.expr_transposed_tensor)
    return self.expr_tensor

  def as_transposed_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each row is one of the embeddings.
    """
    if self.expr_transposed_tensor is None:
      self.expr_transposed_tensor = dy.transpose(self.as_tensor())
    return self.expr_transposed_tensor

  def dim(self) -> tuple:
    """
    Return dimension of the expression sequence

    Returns:
      result of self.as_tensor().dim(), without explicitly constructing that tensor
    """
    if self.has_tensor(): return self.as_tensor().dim()
    else:
      return tuple(list(self[0].dim()[0]) + [len(self)]), self[0].dim()[1]

@xnmt.require_torch
class ExpressionSequenceTorch(BaseExpressionSequence):
  """A class to represent a sequence of expressions.

  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self,
               expr_list: Optional[Sequence[tt.Tensor]] = None,
               expr_tensor: Optional[tt.Tensor] = None,
               expr_transposed_tensor: Optional[tt.Tensor] = None,
               mask: Optional['batchers.Mask'] = None) -> None:
    """Constructor.

    Args:
      expr_list: a python list of expressions
      expr_tensor: a tensor where last dimension are the sequence items
      expr_transposed_tensor: a tensor in transposed form (first dimension are sequence items)
      mask: an optional mask object indicating what positions in a batched tensor should be masked
    Raises:
      valueError: raises an exception if neither expr_list nor expr_tensor are given,
                  or if both have inconsistent length
    """
    super().__init__(expr_list=expr_list,
                     expr_tensor=expr_tensor,
                     expr_transposed_tensor=expr_transposed_tensor,
                     mask=mask,
                     tensor_type=torch.Tensor)

  def __getitem__(self, key):
    """Get a single item.

    Returns:
      sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else:
      if key < 0: key += len(self)
      if self.expr_tensor is not None:
        return torch.index_select(self.expr_tensor, dim=1, index=torch.LongTensor([key]).to(xnmt.device)).squeeze(1)
      else:
        return torch.index_select(self.expr_transposed_tensor, dim=-1, index=torch.LongTensor([key]).to(xnmt.device)).squeeze(-1)

  def as_list(self) -> List[tt.Tensor]:
    """Get a list.

    Returns:
      the whole sequence as a list with each element one of the embeddings.
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return self.expr_list

  def as_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each column is one of the embeddings.
    """
    if self.expr_tensor is None:
      if self.expr_list:
        self.expr_tensor = torch.cat([torch.unsqueeze(e, dim=1) for e in self.expr_list], dim=1)
      else:
        self.expr_tensor = torch.transpose(self.expr_transposed_tensor, 1, 2)
    return self.expr_tensor

  def as_transposed_tensor(self) -> tt.Tensor:
    """Get a tensor.
    Returns:
      the whole sequence as a tensor expression where each row is one of the embeddings.
    """
    if self.expr_transposed_tensor is None:
      self.expr_transposed_tensor = self.as_tensor().transpose(1,2)
    return self.expr_transposed_tensor

  # should get rid of this:
  # def dim(self) -> tuple:
  #   """
  #   Return dimension of the expression sequence
  #
  #   Returns:
  #     result of self.as_tensor().dim(), without explicitly constructing that tensor
  #   """
  #   if self.has_tensor(): return self.as_tensor().dim()
  #   else:
  #     return tuple(list(self[0].dim()[0]) + [len(self)]), self[0].dim()[1]

ExpressionSequence = xnmt.resolve_backend(ExpressionSequenceDynet, ExpressionSequenceTorch)

@xnmt.require_dynet
class LazyNumpyExpressionSequenceDynet(ExpressionSequence):
  """
  This is initialized via numpy arrays, and dynet expressions are only created
  once a consumer requests representation as list or tensor.
  """
  def __init__(self, lazy_data: np.ndarray, mask: Optional['batchers.Mask'] = None) -> None:
    """
    Args:
      lazy_data: numpy array, or Batcher.Batch of numpy arrays
    """
    self.lazy_data = lazy_data
    self.expr_list, self.expr_tensor, self.expr_transposed_tensor = None, None, None
    self.mask = mask
  def sent_len(self):
    if self.expr_list or self.expr_tensor or self.expr_transposed_tensor:
      return super().sent_len()
    else:
      if batchers.is_batched(self.lazy_data):
        return self.lazy_data[0].get_array().shape[1]
      else: return self.lazy_data.get_array().shape[1]
  def batch_size(self):
    if self.expr_list or self.expr_tensor or self.expr_transposed_tensor:
      return super().batch_size()
    else:
      if batchers.is_batched(self.lazy_data):
        return self.lazy_data.batch_size()
      else: return 1
  def __iter__(self):
    if not (self.expr_list or self.expr_tensor):
      self.expr_list = [self[i] for i in range(len(self))]
    return super().__iter__()
  def __getitem__(self, key):
    if self.expr_list or self.expr_tensor:
      return super().__getitem__(key)
    else:
      if batchers.is_batched(self.lazy_data):
        return dy.inputTensor(
          [self.lazy_data[batch].get_array()[:, key] for batch in range(self.lazy_data.batch_size())], batched=True)
      else:
        return dy.inputTensor(self.lazy_data.get_array()[:,key], batched=False)
  def as_tensor(self) -> tt.Tensor:
    if not (self.expr_list or self.expr_tensor):
      if not batchers.is_batched(self.lazy_data):
        raise NotImplementedError()
      array = np.concatenate([d.get_array().reshape(d.get_array().shape + (1,)) for d in self.lazy_data], axis=2)
      self.expr_tensor = dy.inputTensor(array, batched=batchers.is_batched(self.lazy_data))
    return super().as_tensor()

@xnmt.require_torch
class LazyNumpyExpressionSequenceTorch(ExpressionSequence):
  """
  This is initialized via numpy arrays, and torch tensors are only created
  once a consumer requests representation as list or tensor.
  """
  def __init__(self, lazy_data: np.ndarray, mask: Optional['batchers.Mask'] = None) -> None:
    """
    Args:
      lazy_data: numpy array, or Batcher.Batch of numpy arrays
    """
    self.lazy_data = lazy_data
    self.expr_list, self.expr_tensor, self.expr_transposed_tensor = None, None, None
    self.mask = mask
  def sent_len(self):
    if self.expr_list or self.expr_tensor is not None or self.expr_transposed_tensor is not None:
      return super().sent_len()
    else:
      if batchers.is_batched(self.lazy_data):
        return self.lazy_data[0].get_array().shape[0]
      else: return self.lazy_data.get_array().shape[0]

  def batch_size(self):
    if self.expr_list or self.expr_tensor is not None or self.expr_transposed_tensor is not None:
      return super().batch_size()
    else:
      if batchers.is_batched(self.lazy_data):
        return self.lazy_data.batch_size()
      else: return 1

  def __iter__(self):
    if not (self.expr_list or self.expr_tensor):
      self.expr_list = [self[i] for i in range(len(self))]
    return super().__iter__()
  def __getitem__(self, key):
    if self.expr_list or self.expr_tensor:
      return super().__getitem__(key)
    else:
      return torch.tensor(
        np.array([self.lazy_data[batch_i].get_array()[key, :] for batch_i in range(self.lazy_data.batch_size())]),
        device=xnmt.device)
  def as_tensor(self) -> tt.Tensor:
    if not (self.expr_list or self.expr_tensor is not None):
      if not batchers.is_batched(self.lazy_data):
        raise NotImplementedError()
      array = np.concatenate([d.get_array().reshape((1,) + d.get_array().shape) for d in self.lazy_data], axis=0)
      self.expr_tensor = torch.tensor(array, device=xnmt.device)
    return super().as_tensor()
  def as_transposed_tensor(self) -> tt.Tensor:
    if not batchers.is_batched(self.lazy_data):
      raise NotImplementedError()
    array = np.concatenate([d.get_array().T.reshape((1,) + d.get_array().T.shape) for d in self.lazy_data], axis=0)
    self.expr_transposed_tensor = torch.tensor(array, device=xnmt.device)
    return super().as_transposed_tensor()

LazyNumpyExpressionSequence = xnmt.resolve_backend(LazyNumpyExpressionSequenceDynet, LazyNumpyExpressionSequenceTorch)


class BaseReversedExpressionSequence(BaseExpressionSequence):
  def __init__(self, base_expr_seq):
    self.base_expr_seq = base_expr_seq
    self.expr_tensor = None
    self.expr_list = None
    self.expr_transposed_tensor = None
    if base_expr_seq.mask is None:
      self.mask = None
    else:
      self.mask = base_expr_seq.mask.reversed()

  def sent_len(self):
    return self.base_expr_seq.sent_len()

  def batch_size(self):
    return self.base_expr_seq.batch_size()

  def __iter__(self):
    if self.expr_list is None:
      self.expr_list = list(reversed(self.base_expr_seq.as_list()))
    return iter(self.expr_list)

  def __getitem__(self, key):
    return self.base_expr_seq[len(self) - key - 1]

  def as_list(self) -> List[tt.Tensor]:
    if self.expr_list is None:
      self.expr_list = list(reversed(self.base_expr_seq.as_list()))
    return self.expr_list

  def has_list(self) -> bool:
    return self.base_expr_seq.has_list()
  def has_tensor(self) -> bool:
    return self.expr_tensor is not None
  def dim(self):
    return self.base_expr_seq.dim()

@xnmt.require_dynet
class ReversedExpressionSequenceDynet(BaseReversedExpressionSequence, ExpressionSequenceDynet):
  """
  A reversed expression sequences, where expressions are created in a lazy fashion
  """
  def __init__(self, base_expr_seq):
    super().__init__(base_expr_seq=base_expr_seq)

  def as_tensor(self) -> tt.Tensor:
    # note: this is quite memory hungry and should be avoided if possible
    if self.expr_tensor is None:
      if self.expr_list is None:
        self.expr_list = list(reversed(self.base_expr_seq.as_list()))
      self.expr_tensor = dy.concatenate_cols(self.expr_list)
    return self.expr_tensor

@xnmt.require_torch
class ReversedExpressionSequenceTorch(BaseReversedExpressionSequence, ExpressionSequenceTorch):
  """
  A reversed expression sequences, where expressions are created in a lazy fashion
  """
  def __init__(self, base_expr_seq):
    super().__init__(base_expr_seq=base_expr_seq)

  def as_tensor(self) -> tt.Tensor:
    # note: this is quite memory hungry and should be avoided if possible
    if self.expr_tensor is None:
      self.expr_tensor = torch.flip(self.base_expr_seq.as_tensor(), [1])
    return self.expr_tensor

ReversedExpressionSequence = xnmt.resolve_backend(ReversedExpressionSequenceDynet, ReversedExpressionSequenceTorch)


class CompoundSeqExpression(object):
  """ A class that represent a list of Expression Sequence. """

  def __init__(self, exprseq_list):
    self.exprseq_list = exprseq_list

  def __iter__(self):
    return iter(self.exprseq_list)

  def __getitem__(self, idx):
    return self.exprseq_list[idx]