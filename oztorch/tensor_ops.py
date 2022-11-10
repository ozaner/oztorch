from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
  from .tensor import Tensor

import oztorch
from .storage import Storage

# These ops all autocast variables to floats

def map(fn: Callable[[float], float]) -> Callable[['Tensor'], 'Tensor']:
  def temp(input: 'Tensor') -> 'Tensor':
    num = input.numel()
    out_arr = [0.]*num
    for i in range(num):
      out_arr[i] = fn(input.storage.data[i])
    return oztorch.Tensor(Storage(out_arr, input.size(), input.stride()))
  return temp

