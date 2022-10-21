from typing import Callable
from .commonTyping import ValidPyType

class dtype():
  def __init__(self, name: str, cast_fn: Callable[[ValidPyType], ValidPyType]):
    self.name = name
    self.cast_fn = cast_fn

float64: dtype = dtype("float64", float)
int64: dtype = dtype("int64", int)
bool: dtype = dtype("bool", bool)

default_dtype_list = [bool, int64, float64]