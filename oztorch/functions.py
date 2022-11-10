from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
  from .tensor import Tensor

from .autograd import Context, Function
from .tensor_ops import map

class Neg(Function['Tensor']):
  _neg_map: Callable[['Tensor'],'Tensor'] = map(lambda x: -x)

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor') -> 'Tensor':
    return Neg._neg_map(t1)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> 'Tensor':
    return Neg._neg_map(grad_output)
