import typing
import numpy as np

P = typing.ParamSpec("P")
T = typing.TypeVar("T", bound=typing.Any)


# class Array(np.ndarray[typing.Concatenate[P], T]):
#     ...


class nd(typing.Concatenate[P]):
    ...


Array = np.ndarray[nd[P], T]
"""
>>> x: Array[[int], int] = np.array([1, 2, 3])
"""

x = Array[[int], int]
