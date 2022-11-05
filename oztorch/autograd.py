from typing import Generic, Optional, Sequence, TypeVar, Self
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .function import Function

from dataclasses import dataclass
from abc import abstractmethod


class Variable():
  def __init__(self, requires_grad: bool = False, history: Optional['History[Self]'] = None):
    self._requires_grad = requires_grad
    self._history = history or History() #default = empty history (user created)

  @property
  def requires_grad(self) -> bool:
    """
    False by Default. Can only be manually set for leaf variables.
    
    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html
    """
    return self._requires_grad

  @requires_grad.setter
  def requires_grad(self, x):
    if (self.is_leaf):
      self._requires_grad = x
    else:
      raise Exception("You can only change the requires_grad value for leaf variables.")

  @property
  def is_leaf(self) -> bool:
    """
    The following are leaf `Variables`s:
     - All Variables with `require_grad` set to `False`.
     - For Variables with `require_grad` set to `True`, only those directly created by the user is a leaf. In other words, they are not the result of a `Function` and so their `_history.grad_fn` is `None`.

    PyTorch equivalent: https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html
    """
    return not self.requires_grad or self._history.grad_fn is None

  def detach(self) -> Self:
    return self._copy_wo_grad_info(False, None)

  @abstractmethod
  def _copy_wo_grad_info(self, requires_grad: bool, history: Optional['History[Self]']) -> Self: ...


T = TypeVar('T', bound=Variable)

@dataclass
class Context(Generic[T]):
  """
  `Context` class is used by `Function` to store information
  during the forward pass for the backward pass.
  """
  no_grad: bool = False # whether the tensor was recording grads when called
  saved_values: tuple[T, ...] = ()

  def save_for_backward(self, *values: T):
    "Store the given `values` for use during backpropagation."
    if not self.no_grad: #don't bother saving vals if no_grad enabled
      self.saved_values = values

  @property
  def saved_tensors(self) -> tuple[T, ...]:
    return self.saved_values


@dataclass
class History(Generic[T]):
  """
  `History` stores the history of the `Function` operation that was
  used to construct the current `T`.
  """
  grad_fn: Optional[type['Function[T]']] = None
  ctx: Optional[Context[T]] = None
  inputs: Sequence[T] = ()