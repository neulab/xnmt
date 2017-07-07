
import dynet as dy

class ExpressionSequence():
  """A class to represent a sequence of expressions.
  
  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self, **kwargs):
    """Constructor.

    :param expr_list: a python list of expressions
    :param expr_tensor: a tensor where highest dimension are the sequence items
    :raises valueError:
      raises an exception if neither expr_list nor expr_tensor are given,
      or if both have inconsistent length
    """
    self.expr_list = kwargs.pop('expr_list', None)
    self.expr_tensor = kwargs.pop('expr_tensor', None)
    if not (self.expr_list or self.expr_tensor):
      raise ValueError("must provide expr_list or expr_tensor")
    if self.expr_list and self.expr_tensor:
      if len(self.expr_list) != self.expr_tensor.dim()[0][0]:
        raise ValueError("expr_list and expr_tensor must be of same length")

  def __len__(self):
    """Return length.

    :returns: length of sequence
    """
    if self.expr_list: return len(self.expr_list)
    else: return self.expr_tensor.dim()[0][0]

  def __iter__(self):
    """Return iterator.

    :returns: iterator over the sequence; results in explicit conversion to list
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return iter(self.expr_list)

  def __getitem__(self, key):
    """Get a single item.

    :returns: sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else: return dy.pick(self.expr_tensor, key)

  def as_list(self):
    """Get a list.
    :returns: the whole sequence as a list with each element one of the embeddings.
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return self.expr_list

  def as_tensor(self):
    """Get a tensor.
    :returns: the whole sequence as a tensor expression where each column is one of the embeddings.
    """
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate_cols(self.expr_list)
    return self.expr_tensor
