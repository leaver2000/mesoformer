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
    "LiteralUnit",
]
import os
import sys
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Collection,
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
    TypeGuard,
    Sized,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
    Concatenate,
)

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
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


if TYPE_CHECKING:
    from ._stub_files.literal_unit import LiteralUnit
else:
    LiteralUnit = str

# =====================================================================================================================
Ts = TypeVarTuple("Ts")
T = TypeVar("T")
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)

AnyT = TypeVar("AnyT", bound=Any)
KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
ScalarT = TypeVar("ScalarT", bound=Scalar)

NdAny = NewType("Nd[...]", "Nd[EllipsisType]")  # type: ignore[valid-type]
_NdT = TypeVar("_NdT", bound="Nd")
EnumT = TypeVar("EnumT", bound="PlotEnumProtocol")
FrameT = TypeVar("FrameT", bound="FrameProtocol[Any, Any]")

Number = Union[int, float, bool]
DictStr: TypeAlias = dict[str, AnyT]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = "str | os.PathLike[str]"
PatchSize: TypeAlias = 'int | Literal["upscale", "downscale"]'
Pair: TypeAlias = tuple[AnyT, AnyT]
NestedSequence: TypeAlias = Sequence[Union[AnyT, "NestedSequence[AnyT]"]]

if TYPE_CHECKING:
    import torch

    class _TensorLike(torch.Tensor, Generic[_NdT, AnyT]):
        dtype: AnyT


# =====================================================================================================================
# - Protocols
# =====================================================================================================================
class Indices(Sized, Iterable[T1_co], Protocol[T1_co]):
    ...


class Closeable(Protocol):
    def close(self) -> None:
        ...


class Shaped(Sized, Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...


class FrameProtocol(Shaped, Generic[T1_co, T2_co], Protocol):
    @property
    def columns(self) -> T1_co:
        ...

    @property
    def dtypes(self) -> T2_co:
        ...


class GetItemSliceToSelf(Sized, Protocol[T1_co]):
    def __getitem__(self, idx: slice) -> Sequence[T1_co]:
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


class PlotEnumProtocol(EnumProtocol, Protocol):
    @classmethod
    def map(cls: type[EnumT], __iterable: Iterable[Any] | None = None) -> tuple[EnumT, ...]:
        ...

    @classmethod
    def __iter__(cls) -> Self:
        ...

    @classmethod
    def __next__(cls) -> Self:
        ...


# =====================================================================================================================
class Nd(Generic[Unpack[Ts]]):
    ...


# =====================================================================================================================
N = NewType(":", int)
NAny = NewType("...", Any)
Batch = NewType("Batch", int)
Channel = NewType("Channel", int)
Length = NewType("Length", int)
Width = NewType("Width", int)
Time = NewType("Time", int)
Height = NewType("Height", int)
Array: TypeAlias = np.ndarray[_NdT, np.dtype[AnyT]]


TensorLike: TypeAlias = "Union[_TensorLike[Nd[Unpack[Ts]], torch.dtype], torch.Tensor]"
AnyArray: TypeAlias = "Array[Nd[Unpack[Ts]], Any] | TensorLike[Nd[Unpack[Ts]], torch.dtype]"
TensorF32: TypeAlias = "TensorLike[Nd[Unpack[Ts]], torch.dtype]"
