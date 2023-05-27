# For things like sizes or strides
IntTuple = tuple[int,...]

# # Size could be an int tuple or a single int (implying a rank 1 tensor)
# Size = IntTuple | int

# any python data type that corresponds to a valid tensor type (complex currently not supported)
ValidPyType = bool | int | float

# The type of an array backing a tensor
BackingList = list[bool] | list[int] | list[float]

# ## Defining PyTensor type
# from typing import Sequence, TypeVar
# _T = TypeVar("_T")
# _MultiDimSeq = Sequence[_T] | Sequence["_MultiDimSeq[_T]"] # literal needed to resolve recursive type
# _GenericPyTensor = _T | _MultiDimSeq[_T]
# PyTensor = _GenericPyTensor[ValidPyType] # n-D sequence w/ mixed entries