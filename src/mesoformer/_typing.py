from __future__ import annotations

__all__ = [
    "Nd",
    "Array",
    "ListLike",
    "ArrayLike",
    "AnyArrayLike",
    "NestedSequence",
    "TensorLike",
]
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch
    from numpy._typing._nested_sequence import _NestedSequence as NestedSequence
    from pandas.core.arrays.base import ExtensionArray

    _P = typing.ParamSpec("_P")
    _T = typing.TypeVar("_T", bound=typing.Any)
    _T_co = typing.TypeVar("_T_co", bound=typing.Any, covariant=True)
    _T_contra = typing.TypeVar("_T_contra", bound=typing.Any, contravariant=True)
    _NumpyT_co = typing.TypeVar("_NumpyT_co", covariant=True, bound=np.generic)

    class Nd(typing.Concatenate[_P]):
        ...

    Array = np.ndarray[Nd[_P], np.dtype[_NumpyT_co]]
    """>>> x: Array[[int,int], np.int_] = np.array([[1, 2, 3]]) # Array[(int,int), int]"""
    NDArray = Array[[...], _NumpyT_co]
    ArrayLike: typing.TypeAlias = typing.Union[ExtensionArray, Array[..., _NumpyT_co]]
    AnyArrayLike: typing.TypeAlias = typing.Union[ArrayLike[_T], pd.Index[_T], pd.Series[_T]]
    List = list[_T | typing.Any]
    ListLike = typing.Union[AnyArrayLike[_T], List[_T]]
    TensorLike = typing.Union[Array[_P, _T_co], NDArray[_T_co], torch.Tensor]


else:
    import numpy.typing as npt

    NestedSequence = typing.List  # NestedSequence[int]
    Nd = typing.Tuple  # Nd[int, int, ...]
    Array = typing.Callable  # Array[[int,int], np.int_]
    NDArray = npt.NDArray
    ArrayLike = typing.List  # ArrayLike[int]
    AnyArrayLike = typing.List  # AnyArrayLike[int]
    List = typing.List
    ListLike = typing.List  # ListLike[int]
    TensorLike = typing.Callable  # TensorLike[[int,int], torch.int_]
