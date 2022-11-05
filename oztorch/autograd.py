from typing import Optional, Sequence, Type
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .tensor import Tensor
  from .function import Function

from dataclasses import dataclass


@dataclass
class Context:
  """
  `Context` class is used by `Function` to store information
  during the forward pass for the backward pass.
  """
  no_grad: bool = False # whether the tensor was recording grads when called
  saved_values: tuple['Tensor', ...] = ()

  def save_for_backward(self, *values: 'Tensor') -> None:
    "Store the given `values` for use during backpropagation."
    if not self.no_grad: #don't bother saving vals if no_grad enabled
      self.saved_values = values

  @property
  def saved_tensors(self) -> tuple['Tensor', ...]:
    return self.saved_values


@dataclass
class History:
  """
  `History` stores the history of the `Function` operation that was
  used to construct the current `Tensor`.
  """
  grad_fn: Optional[Type['Function']] = None
  ctx: Optional[Context] = None
  inputs: Sequence['Tensor'] = ()