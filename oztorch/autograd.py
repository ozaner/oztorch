from typing import Generic, Iterable, Optional, Sequence, TypeVar, Self

from dataclasses import dataclass
from abc import abstractmethod


class Variable():

  # every variable of a certain type should have a unique identifier.
  unique_id: int

  def __init__(self, requires_grad: bool = False, history: Optional['History[Self]'] = None):
    self._requires_grad = requires_grad
    self._history = history or History() #default = empty history (user created)
    self.grad = None #grad unset by default

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

  def accumulate_derivative(self, x: Self) -> None:
    """
    Add `x` to the the derivative accumulated on this variable.
    Should only be called during autodifferentiation on leaf variables.

    Args:
        x : value to be accumulated
    """
    assert self.is_leaf, "Only leaf variables can have derivatives."
    if self.grad is None:
      self.grad = self._create_empty_grad()
    self.grad += x

  def topological_sort(self) -> Iterable[Self]:
    """
    Computes the topological order of the computation graph with this variable as the root.

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    result: list[Variable] = []
    seen = set()

    def helper(node: Variable) -> None:
      if node.unique_id in seen or not node.requires_grad:
        return #skip already processed vars or vars that don't 'require_grad'
      if not node.is_leaf:
        for neighbor in node._history.inputs:
          if node.requires_grad:
            helper(neighbor)
      seen.add(node.unique_id)
      result.insert(0, node)

    helper(self) #call helper on root
    return result

  def backward(self, root_der: Self):
    """
    Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes.
    Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Args:
        root_der: Its derivative that we want to propagate backward to the leaves.
    """
    queue = self.topological_sort()
    var_dict = {}
    var_dict[self.unique_id] = [self, root_der]  # add root

    for node in queue:
      if node.is_leaf:
        continue # leafs don't need to be chained

      # topos ensures parents have been processed and are in dict
      for var, der in node.chain_rule(var_dict[node.unique_id][1]):
        if (var.unique_id in var_dict):  # already in dict -> another partial derivative
          var_dict[var.unique_id][1] += der
        else:
          var_dict[var.unique_id] = [var, der]

        if var.is_leaf:
          var.accumulate_derivative(der)

  def chain_rule(self, d_output: Self) -> Iterable[tuple[Self, Self]]:
    h = self._history
    assert h is not None
    assert h.grad_fn is not None
    assert h.ctx is not None

    x = h.grad_fn.backward(h.ctx, d_output)
    if not isinstance(x, tuple):
      x = (x,) #some backwards funcs only ouput a single var
    assert len(x) == len(h.inputs), f"Bug in function {h.grad_fn}"
    # return [
    #   (inp, inp.expand(d_in))
    #   for inp, d_in in zip(h.inputs, x)
    # ]
    return zip(h.inputs, x)

  @abstractmethod
  def _copy_wo_grad_info(self, requires_grad: bool, history: Optional['History[Self]']) -> Self: ...

  @abstractmethod
  def _create_empty_grad(self) -> Self: ...

  @abstractmethod
  def __add__(self, b: Self): ...


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


class Function(Generic[T]):
  @classmethod
  @abstractmethod
  def forward(cls, ctx: Context, *inps: T) -> T:
    ...

  @classmethod
  @abstractmethod
  def backward(cls, ctx: Context, *inps: T) -> T | tuple[T, ...]:
    ...

  @classmethod
  def apply(cls, *vals: T) -> T:

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
    return out._copy_wo_grad_info(need_grad, history)