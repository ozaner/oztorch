from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
  from .tensor import Tensor

import oztorch
from .storage import broadcast_shape, contiguous_raw_to_index, broadcast_index

# These ops all autocast variables to floats

def map(fn: Callable[[float], float]) -> Callable[['Tensor'], 'Tensor']:
  def fn_tensor(t: 'Tensor') -> 'Tensor':
    out = oztorch.zeros(t.size())
    for i in range(t.numel()):
      out.storage.data[i] = fn(t.storage.data[i])
    return out
  return fn_tensor

def zip(fn: Callable[[float, float], float]) -> Callable[['Tensor', 'Tensor'], 'Tensor']:
  def fn_tensor(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    out_shape = broadcast_shape(t1.size(), t2.size())
    out = oztorch.zeros(out_shape)  
    for i in range(out.numel()):
      index = contiguous_raw_to_index(i, out_shape)
      index1 = broadcast_index(index, out_shape, t1.size())
      index2 = broadcast_index(index, out_shape, t2.size())
      out[index] = fn(t1[index1], t2[index2])
    return out
  return fn_tensor

def reduce(fn: Callable[[float, float], float]) -> Callable[['Tensor'], 'Tensor']:
  