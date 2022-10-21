# For things like sizes or strides
IntTuple = tuple[int,...]

# any python data type that corresponds to a valid tensor type (complex currently not supported)
ValidPyType = bool | int | float

# The type of an array backing a tensor
BackingList = list[bool] | list[int] | list[float]
