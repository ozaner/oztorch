from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
  from .tensor import Tensor

from math import log

from .autograd import Context, Function
from .tensor_ops import map, zip

class Neg(Function['Tensor']):
  _neg_map: Callable[['Tensor'],'Tensor'] = map(lambda x: -x)

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor') -> 'Tensor':
    return Neg._neg_map(t1)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> 'Tensor':
    return Neg._neg_map(grad_output)

class Inv(Function):
  _inv_map: Callable[['Tensor'],'Tensor'] = map(lambda x: 1/x)
  _inv_back_zip: Callable[['Tensor','Tensor'],'Tensor'] = zip(lambda x, d: -d/(x*x))

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor') -> 'Tensor':
    ctx.save_for_backward(t1)
    return Inv._inv_map(t1)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> 'Tensor':
    (t1,) = ctx.saved_values
    return Inv._inv_back_zip(t1, grad_output)

class Log(Function['Tensor']):
  _log_map: Callable[['Tensor'],'Tensor'] = map(lambda x: log(x))
  _log_back_zip: Callable[['Tensor','Tensor'],'Tensor'] = zip(lambda x, d: d*(1/x))

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor') -> 'Tensor':
    ctx.save_for_backward(t1)
    return Log._log_map(t1)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> 'Tensor':
    (t1,) = ctx.saved_values
    return Log._log_back_zip(t1, grad_output)

class Add(Function):
  _add_zip: Callable[['Tensor','Tensor'],'Tensor'] = zip(lambda x,y: x+y)

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    return Add._add_zip(t1, t2)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> tuple['Tensor', 'Tensor']:
    return grad_output, grad_output

class Mul(Function):
  _mul_zip: Callable[['Tensor','Tensor'],'Tensor'] = zip(lambda x,y: x*y)

  @staticmethod
  def forward(ctx: Context, t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    ctx.save_for_backward(t1, t2)
    return Mul._mul_zip(t1, t2)

  @staticmethod
  def backward(ctx: Context, grad_output: 'Tensor') -> tuple['Tensor', 'Tensor']:
    (t1, t2) = ctx.saved_values
    # return grad_output * t2, grad_output * t1 #goes through the trouble of calling `apply`
    return Mul._mul_zip(grad_output, t2), Mul._mul_zip(grad_output, t1)