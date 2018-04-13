import os
from typing import TypeVar, Sequence, Union, Dict,List
T = TypeVar('T')

OneOrSeveral = Union[T,Sequence[T]]

YamlSerializable=Union[None,bool,int,float,'Serializable',List['YamlSerializable'],Dict[str,'YamlSerializable']]

def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise
