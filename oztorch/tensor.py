from typing import Optional, Sequence, overload
from .commonTyping import IntTuple, ValidPyType
from .dtype import dtype

from math import prod

from .storage import Storage
from .autograd import History, Variable


### -----------------------------
## Tensor class
### -----------------------------
class Tensor(Variable):
  def __init__(self, storage: Storage, /, requires_grad: bool = False, history: Optional[History] = None):
    # Variable stuff
    super().__init__(requires_grad, history)

    # init backing storage
    self.storage = storage


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
      return self.storage.size
    else:
      return self.storage.size[dim]
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
    return self.storage.numel()

  # Returns the strides of the tensor
  def stride(self) -> IntTuple:
    return self.storage.stride
  
  def __getitem__(self, key: IntTuple | int) -> ValidPyType:
    return self.storage[key]

  def __setitem__(self, key: IntTuple | int, val: ValidPyType):
    self.storage[key] = val

  def __repr__(self) -> str:
    return self.storage.to_string()


  ### -----------------------------
  ## Variable stuff (autograd)
  ### -----------------------------
  def _copy_wo_grad_info(self, requires_grad: bool, history: Optional[History]) -> 'Tensor':
    """
    Creates a copy of this `Tensor` (with the same underlying storage)
    without a history, and with `requires_grad` set to `False`
    """
    return Tensor(self.storage, requires_grad=requires_grad, history=history)


### -----------------------------
## Private methods
### -----------------------------
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

def _clean_PyTensor(data: Sequence | ValidPyType) -> tuple[list[ValidPyType], IntTuple]:
  size = []
  flattened_data = _clean_PyTensor_helper(data, 0, size)
  if size[-1] == -1:
    size.pop() #remove single element psuedo-dimension
  return flattened_data, tuple(size)


### -----------------------------
## Static methods
### -----------------------------
#creates a Tensor using an n-dim python sequence (of valid dtypes)
def tensor(data: Sequence | ValidPyType, /, dtype: Optional[dtype] = None, requires_grad: bool = False) -> Tensor:
  flattened_data, size = _clean_PyTensor(data)
  return Tensor(Storage(flattened_data, size, dtype=dtype), requires_grad = requires_grad)

def full(size: IntTuple, fill_value: ValidPyType, /, dtype: Optional[dtype] = None, requires_grad: bool = False) -> Tensor:
  numel = prod(size)
  data = [fill_value]*numel
  return Tensor(Storage(data, size, dtype=dtype), requires_grad = requires_grad)

def zeros(size: IntTuple, /, dtype: Optional[dtype] = None, requires_grad: bool = False) -> Tensor:
  return full(size, 0.0, dtype=dtype, requires_grad = requires_grad)

def ones(size: IntTuple, /, dtype: Optional[dtype] = None, requires_grad: bool = False) -> Tensor:
  return full(size, 1.0, dtype=dtype, requires_grad = requires_grad)