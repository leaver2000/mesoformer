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
]
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
from numpy.typing import ArrayLike
from numpy.typing import NDArray as _NDArray
from pandas._typing import Scalar

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

P = ParamSpec("P")
if TYPE_CHECKING:
    from numpy._typing._nested_sequence import _NestedSequence as NestedSequence

    from ._typing._tensor import TensorLike  # pyright: ignore

    class nd(Concatenate[P]):
        ...

else:
    TensorLike = Callable
    NestedSequence: TypeAlias = "Sequence[T | NestedSequence[T]]"
    nd = tuple
# =====================================================================================================================
T = TypeVar("T", bound=Any)
T_co = TypeVar("T_co", bound=Any, covariant=True)
T_contra = TypeVar("T_contra", bound=Any, contravariant=True)
Ts = TypeVarTuple("Ts")

NDArray: TypeAlias = _NDArray[T_co] | _NDArray[Any]

EnumT = TypeVar("EnumT", bound="EnumProtocol")
Boolean: TypeAlias = bool | np.bool_
Number: TypeAlias = Union[int, float, np.number]
SequenceLike: TypeAlias = "NestedSequence[T] | TensorLike[..., T] | NDArray[T]"


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


class EnumProtocol(Protocol[T]):
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


N = NewType("N", int)


Array = np.ndarray[nd[P], T_contra] | NDArray[T_contra]
""">>> x: Array[[N], np.int_] = np.array([1, 2, 3]) # Array[(N), int]"""
