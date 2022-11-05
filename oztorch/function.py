from abc import abstractmethod

from .tensor import Tensor
from .autograd import Context, History


class Function:
  @classmethod
  @abstractmethod
  def forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
    ...

  @classmethod
  @abstractmethod
  def backward(cls, ctx: Context, *inps: Tensor) -> tuple[Tensor, ...]:
    ...

  @classmethod
  def apply(cls, *vals: Tensor) -> Tensor:

    # determines if this function call needs to be added to the graph
    # i.e. if any of its inputs requires grad
    need_grad = any([v.requires_grad for v in vals])

    # Create the context.
    # - Specify no_grad so that forward pass can skip saving if unneeded
    ctx = Context(no_grad = not need_grad)

    # if we didn't detach the vals before calling forward,
    # then the computation graph would include the intermediate function calls inside `forward`,
    # even though `backward` should already account for these
    clean_vals = [v.detach() for v in vals]

    # Call forward with the detached inputs.
    out = cls.forward(ctx, *clean_vals)

    # Initalize history, if need_grad.
    # Else leave blank (tensor will be a leaf, unknowing of its history)
    history = History(cls, ctx, vals) if need_grad else None

    # Create a new tensor from the out and history.
    return Tensor(out._storage, out._size, stride = out._stride, requires_grad = need_grad, history = history)
