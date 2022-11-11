from itertools import zip_longest
from typing import Optional

from math import prod

from .dtype import dtype, default_dtype_list
from .commonTyping import IntTuple, ValidPyType

class Storage():
  def __init__(self, data: list[ValidPyType], size: IntTuple, /, stride: Optional[IntTuple] = None, dtype: Optional[dtype] = None):
    # set backing fields
    self.size = size
    self.stride = stride or _calculate_standard_stride(size)
    self.data = data

    #cast storage to dtype (or minimal dtype)
    _cast_list_in_place(self.data, dtype)

  # Returns number elements in the logical tensor (not the underlying storage, i.e. shape matters)
  def numel(self) -> int:
    return prod(self.size)

  def index_to_raw(self, index: IntTuple) -> int:
    return sum(i*j for i,j in zip(index, self.stride))

  def _preprocess_indexing(self, key: IntTuple | int) -> IntTuple:
    # cast single int to IntTuple
    if isinstance(key, int):
      key = (key,)

    # check if key is valid
    if len(key) != len(self.size):
      raise Exception("Tensors currently only support indexing individual elements.")

    return key

  def __getitem__(self, key: IntTuple | int) -> ValidPyType:
    key = self._preprocess_indexing(key)
    return self.data[self.index_to_raw(key)]

  def __setitem__(self, key: IntTuple | int, val: ValidPyType):
    key = self._preprocess_indexing(key)
    self.data[self.index_to_raw(key)] = val

  def to_string(self) -> str:
    s = ""
    for raw_index in range(self.numel()):
      index = contiguous_raw_to_index(raw_index, self.size)
      l = ""
      for i in range(len(index) - 1, -1, -1):
        if index[i] == 0:
          l = "\n%s[" % ("\t" * i) + l
        else:
          break
      s += l
      v = self[index]
      s += f"{v}"
      l = ""
      for i in range(len(index) - 1, -1, -1):
        if index[i] == self.size[i] - 1:
          l += "]"
        else:
          break
      if l:
        s += l
      else:
        s += " "
    return f"tensor({s})"


### -----------------------------
## Public Helper Functions
### -----------------------------
def broadcast_shape(s1: IntTuple, s2: IntTuple) -> IntTuple:
  size: list[int] = []
  for a, b in zip_longest(reversed(s1), reversed(s2)):
    dim = 0
    if a is None:
      dim = b
    elif b is None:
      dim = a
    elif a == 1:
      dim = b
    elif b == 1:
      dim = a
    elif a == b:
      dim = a
    else:
      raise Exception(f"Shape {s1} and shape {s2} cannot be broadcasted.")
    size.append(dim)
  return tuple(reversed(size))

def broadcast_index(b_index: IntTuple, broadcasted_s: IntTuple, s: IntTuple) -> IntTuple:
  out_index = [0]*len(s)
  missing_dims = len(broadcasted_s) - len(s)
  for i, d in enumerate(s):
    if d > 1:
      out_index[i] = b_index[i + missing_dims]
    else:
      out_index[i] = 0
  return tuple(out_index)

def contiguous_raw_to_index(ordinal: int, size: IntTuple) -> IntTuple:
  output: list[int] = [0]*len(size)
  for i, base in reversed(list(enumerate(size))):
      output[i] = ordinal % base
      ordinal = ordinal // base
  return tuple(output)


### -----------------------------
## Private Helper Functions
### -----------------------------
def _calculate_standard_stride(size: IntTuple) -> IntTuple:
  ls = [0]*len(size)
  product = 1
  for i, s in reversed(list(enumerate(size))):
    ls[i] = product
    product*=s
  return tuple(ls)

def _cast_list_in_place(storage: list[ValidPyType], dtype: Optional[dtype]):
  # determine minimal numeric type
  if dtype is None:
    highest_rank = -1
    for i in range(len(storage)):
      if isinstance(storage[i], bool) and highest_rank < 0:
        highest_rank = 0
      elif isinstance(storage[i], int) and highest_rank < 1:
        highest_rank = 1
      elif isinstance(storage[i], float):
        highest_rank = 2
      
    if highest_rank == -1:
      highest_rank = 2 #default is float64
    dtype = default_dtype_list[highest_rank]
  
  # cast all values
  for i in range(len(storage)):
    storage[i] = dtype.cast_fn(storage[i])