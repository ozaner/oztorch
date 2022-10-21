from math import prod
from typing import Optional, Sequence, overload

from .commonTyping import IntTuple, ValidPyType
from .dtype import dtype
import oztorch


### -----------------------------
## Tensor class
### -----------------------------
class Tensor:
  def __init__(self, storage: list[ValidPyType], size: IntTuple, /, stride: Optional[IntTuple] = None, dtype: Optional[dtype] = None):
    # set backing fields    
    self._size = size
    self._stride = stride or _calculate_standard_strides(size)
    self._storage = storage

    #cast storage to dtype (or minimal dtype)
    _cast_list_in_place(self._storage, dtype)

  ## Size method & overloads ------------------
  @overload
  def size(self) -> IntTuple:
    """
    Returns a tuple of the sizes of each dimension.

    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.Tensor.size.html,
    (Note that an `IntTuple` is returned rather than a dedicated `Size` class).
    """
    ...
  @overload
  def size(self, dim: int) -> int:
    """
    Returns the size of the dim-th dimension.

    Args:
      dim (int): The index of the specific dimension whose size is to be returned.

    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.Tensor.size.html
    """
    ...

  def size(self, dim: Optional[int] = None) -> IntTuple | int:
    if dim is None:
      return self._size
    else:
      return self._size[dim]
  ## ---------------------------------------

  def dim(self) -> int:
    """
    Returns the tensor's rank. In otherwords, the number of dimensions in the tensor (i.e. the length of `size()`).

    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html
    """
    return len(self.size())

  # Returns number elements in the logical tensor (not the underlying storage, i.e. shape matters)
  def numel(self) -> int:
    """
    Returns the number of elements in the tensor (i.e. the product of `size()`).

    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.numel.html
    """
    return prod(self.size())

  # Returns the strides of the tensor
  def stride(self) -> IntTuple:
    return self._stride


### -----------------------------
## Private methods
### -----------------------------
def _calculate_standard_strides(size: IntTuple) -> IntTuple:
  ls = [1]
  product = 1
  for i in range(len(size)):
    product*=size[i]
    ls.insert(0, product)
  return tuple(ls)

def _clean_PyTensor_helper(data: Sequence | ValidPyType, dim: int, sizes: list) -> list[ValidPyType]:
  if isinstance(data, Sequence):
    size = len(data)
  else:
    size = -1 #leaf value

  # updates sizes & check for raggedness
  if dim < len(sizes):
    if sizes[dim] != size: #doesn't match others in dimension
      raise Exception(f"Invalid Tensor: ragged in dimension {dim}.")
  else: #first in this dimension
    assert len(sizes) == dim #TODO: test and remove this
    sizes.append(size) #sizes should have length dim at this point
  
  # continue
  if isinstance(data, Sequence):
    return [val for i in range(size) for val in _clean_PyTensor_helper(data[i], dim+1, sizes)]
  else:
    return [data]

def _clean_pyTensor(data: Sequence | ValidPyType) -> tuple[list[ValidPyType], IntTuple]:
  size = []
  flattened_data = _clean_PyTensor_helper(data, 0, size)
  if size[-1] == -1:
    size.pop() #remove single element psuedo-dimension
  return flattened_data, tuple(size)

def _cast_list_in_place(storage: list[ValidPyType], dtype: Optional[dtype]):
  # determine minimal numeric type
  if dtype is None:
    highest_rank = -1
    for i in range(len(storage)):
      if isinstance(storage[i], bool) and highest_rank < 0:
        highest_rank = 0
      elif isinstance(storage[i], int) and highest_rank < 1:
        highest_rank = 1
      else: #must be float
        highest_rank = 2
    
    if highest_rank == -1:
      highest_rank = 2 #default is float64
    dtype = oztorch.default_dtype_list[highest_rank]
  
  # cast all values
  for i in range(len(storage)):
    storage[i] = dtype.cast_fn(storage[i])


### -----------------------------
## Static methods
### -----------------------------
#creates a Tensor using an n-dim python sequence (of valid dtypes)
def tensor(data: Sequence | ValidPyType, /, dtype: Optional[dtype] = None) -> Tensor:
  flattened_data, size = _clean_pyTensor(data)
  return Tensor(flattened_data, size, dtype=dtype)

# Returns a tensor of full zeros of given size
def full(size: IntTuple, fill_value: ValidPyType, /, dtype: Optional[dtype] = None) -> Tensor:
  numel = prod(size)
  data = [fill_value]*numel
  return Tensor(data, size, dtype=dtype)

def zeros(size: IntTuple, /, dtype: Optional[dtype] = None) -> Tensor:
  return full(size, 0.0, dtype=dtype)

def ones(size: IntTuple, /, dtype: Optional[dtype] = None) -> Tensor:
  return full(size, 1.0, dtype=dtype)