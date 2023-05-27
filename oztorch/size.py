class Size(tuple[int, ...]):
  # need to override __new__ (not __init__) because tuple is immutable
  def __new__(cls, *dims: int) -> 'Size': # forward declaration
    if any(d < 0 for d in dims):
      raise Exception('Tensor Size components must be nonnegative.')
    return tuple.__new__(Size, dims)