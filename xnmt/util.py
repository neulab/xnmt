import os
from typing import TypeVar, Sequence, Union
T = TypeVar('T')

OneOrSeveral = Union[T,Sequence[T]]

def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise
