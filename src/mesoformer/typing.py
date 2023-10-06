"""
type annotations and compatibility for python 3.9+ 

"""
from __future__ import annotations

__all__ = [
    "TYPE_CHECKING",
    "Any",
    "Callable",
    "Generic",
    "Mapping",
    "cast",
    "Iterator",
    "Final",
    "TypeAlias",
    "TypedDict",
    "Self",
    "TypeVarTuple",
    "Unpack",
    "ParamSpec",
    "overload",
    "Annotated",
    "Collection",
    "EllipsisType",
    "NDArray",
    "ArrayLike",
    "Concatenate",
    "TypeGuard",
    "TensorLike",
    "Scalar",
    "Hashable",
    "Literal",
    "AnyArrayLike",
    "Array",
    "ArrayLike",
    "ListLike",
    "Nd",
    "NestedSequence",
    "TensorLike",
    "Sequence",
]
import datetime
import os
import sys
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Collection,
    Concatenate,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NewType,
    Protocol,
    Sequence,
    Sized,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from pandas._typing import Scalar

from ._typing import (
    AnyArrayLike,
    Array,
    ArrayLike,
    ListLike,
    Nd,
    NDArray,
    NestedSequence,
    TensorLike,
)

if sys.version_info <= (3, 9):
    from typing_extensions import ParamSpec, Self, TypeAlias, TypeVarTuple, Unpack

    EllipsisType: TypeAlias = "ellipsis"  # noqa
elif sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import ParamSpec, TypeAlias

    from typing_extensions import Self, TypeVarTuple, Unpack
else:
    from types import EllipsisType
    from typing import ParamSpec, Self, TypeAlias, TypeVarTuple, Unpack

# =====================================================================================================================
N = NewType("N", Any)
P = ParamSpec("P")
T = TypeVar("T", bound=Any)
T_co = TypeVar("T_co", bound=Any, covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


EnumT = TypeVar("EnumT", bound="EnumType")
HashKeyT = TypeVar("HashKeyT", bound="HashKeyLike")
Number: TypeAlias = int | float | np.number[Any]
Boolean: TypeAlias = bool | np.bool_
NumberT = TypeVar("NumberT", int, float, np.number[Any])


PythonScalar = Union[str, float, bool]
DatetimeLikeScalar = Union["pd.Period", "pd.Timestamp", "pd.Timedelta"]
PandasScalar = Union["pd.Period", "pd.Timestamp", "pd.Timedelta", "pd.Interval"]
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime.date]

Pair: TypeAlias = tuple[T, T]
DictStr: TypeAlias = dict[str, T]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = "str | os.PathLike[str]"

# =====================================================================================================================
# - Protocols
# =====================================================================================================================


class Indices(Sized, Iterable[T_co], Protocol[T_co]):
    ...


class Closeable(Protocol):
    def close(self) -> None:
        ...


class Shaped(Sized, Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...


class EnumType(Protocol[T]):
    value: T
    __iter__: Callable[..., Iterable[Self]]

    @classmethod
    def __len__(cls) -> int:
        ...

    @classmethod
    def __next__(cls) -> Self:
        ...

    @classmethod
    def __getitem__(cls, name: str) -> Self:
        ...

    @classmethod
    def __call__(cls, value: Any) -> Self:
        ...


class Comparable(Protocol[T_contra]):
    def __ge__(self, __: T_contra) -> bool:
        ...

    def __gt__(self, __: T_contra) -> bool:
        ...

    def __le__(self, __: T_contra) -> bool:
        ...

    def __lt__(self, __: T_contra) -> bool:
        ...


class HashKeyLike(Comparable[T_contra], Hashable, Protocol[T_contra]):
    ...
