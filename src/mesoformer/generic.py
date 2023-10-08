"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import enum
import queue
import random
import threading

import numpy as np
from torch.utils.data import IterableDataset
import pandas as pd
from .typing import (
    Any,
    ArrayLike,
    Concatenate,
    DictStrAny,
    # EnumT,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    N,
    NDArray,
    NestedSequence,
    ParamSpec,
    Self,
    Sequence,
    T,
    TypeVar,
    TypeAlias,
    overload,
    # EnumT,
)
from .utils import indent_kv, squish_map

from typing import Callable

# _MetaCls: TypeAlias = "EnumMetaBase"
# MetaT = TypeVar("MetaT", bound=_MetaCls)
from typing import Any
from typing import Concatenate

K = TypeVar("K", bound=Hashable)
S = TypeVar("S")
P = ParamSpec("P")

import enum
import pandas as pd
import abc

# from typing import Generic, TypeVar, Any, Callable, Iterable, Iterator, Hashable, Union, overload, TypeAlias, Mapping


Aliases = Mapping[str, list[str]]
_EnumNames: TypeAlias = str | Iterable[str] | Iterable[Iterable[str | Any]] | Mapping[str, Any]

EnumT = TypeVar("EnumT", bound=enum.Enum)

from typing import Protocol, TypeVar, ClassVar

_T = TypeVar("_T", bound=Hashable, covariant=True)

EnumMember = TypeVar("EnumMember", bound="_EnumSelf")


class _EnumSelf(Hashable, Protocol[_T]):  # type: ignore
    _names: ClassVar[pd.Index[str]]
    _values: ClassVar[pd.Series[_T]]
    _member_map_: ClassVar[pd.DataFrame]

    @classmethod
    def __getitem__(cls, __names: ListLike[bool] | ListLike[Hashable] | bool | Hashable) -> Self | list[Self]:
        ...

    @classmethod
    def is_in(cls, x: Hashable | Iterable[Hashable]) -> pd.Series[bool]:
        ...

    @classmethod
    def intersection(cls, __x: Hashable | Iterable[Hashable]) -> list[Self]:
        ...

    @classmethod
    def difference(cls, __x: Hashable | Iterable[Hashable]) -> list[Self]:
        ...

    # name: str
    # value: T_contra
    # __iter__: Callable[..., Iterable[Self]]

    # @classmethod
    # def __len__(cls) -> int:
    #     ...

    # @classmethod
    # def __next__(cls) -> Self:
    #     ...

    # @classmethod
    # def __getitem__(cls, name: str) -> Self:
    #     ...
    # @property
    # @classmethod
    # def _values(cls) -> pd.Series[_T]:
    #     ...

    # def __call__(cls, value: Any) -> Self:
    #     ...


from .typing import ListLike

Items = ListLike[bool] | ListLike[Hashable] | bool | Hashable

EnumCls = TypeVar("EnumCls", bound=enum.EnumMeta)


class GenericEnumMeta(enum.EnumMeta, Generic[EnumT]):
    def __class_getitem__(cls, __x: Any) -> TypeAlias:
        return cls


class PandasEnumMeta(GenericEnumMeta[EnumT]):
    __iter__: Callable[..., Iterator[EnumT]]  # type: ignore[assignment]
    _member_map_: pd.DataFrame

    def __new__(
        cls,
        name: str,
        bases: tuple[Any, ...],
        cls_dict: enum._EnumDict,
        aliases: Aliases | None = None,
    ):
        obj = super().__new__(cls, name, bases, cls_dict)

        if aliases is None:
            aliases = obj._get_aliases()
        obj._member_map_ = df = pd.DataFrame.from_dict(
            {k: [v, *aliases.get(k, [])] for k, v in obj._member_map_.items()}, orient="index"
        )
        df.index.name = cls.__name__
        if df.duplicated().any():
            raise ValueError(f"Duplicate values found in {cls.__name__}._member_map_.")

        return obj

    @overload  # type: ignore
    def __getitem__(cls: type[EnumMember], __names: str) -> EnumMember:
        ...

    @overload
    def __getitem__(cls: type[EnumMember], __names: ListLike[bool | Hashable]) -> list[EnumMember]:
        ...

    def __getitem__(cls: type[EnumMember], __names: str | ListLike[bool | Hashable]) -> EnumMember | list[EnumMember]:
        x = cls._values[__names]
        if isinstance(x, pd.Series):
            return x.to_list()
        return x

    @overload  # type: ignore
    def __call__(cls: type[EnumMember], __items: str | Hashable) -> EnumMember:
        ...

    @overload  # type: ignore
    def __call__(cls: type[EnumMember], __items: Iterable[Hashable]) -> list[EnumMember]:
        ...

    def __call__(cls: type[EnumMember], __items: str | Iterable[Hashable] | Hashable) -> EnumMember | list[EnumMember]:
        if isinstance(__items, str):
            (name,) = cls._names[(cls._member_map_ == __items).any(axis=1)]
            return cls._values[name]
        return cls.intersection(__items)

    def to_frame(cls) -> pd.DataFrame:
        return cls._member_map_

    def to_series(cls):
        return cls._values

    @property
    def _names(cls) -> pd.Index[str]:
        return cls._member_map_.index

    @property
    def _values(cls) -> pd.Series[EnumT]:
        return cls._member_map_.iloc[:, 0]

    def to_list(cls) -> list[EnumT]:
        return cls._values.to_list()

    def _get_aliases(cls) -> Mapping[str, list[str]]:
        return {}

    def is_in(cls, __x: Hashable | Iterable[Hashable]) -> pd.Series[bool]:
        if isinstance(__x, str):
            __x = [__x]
        return cls.to_frame().isin(__x).any(axis=1)

    def difference(cls: type[EnumMember], __x: Hashable | Iterable[Hashable]):
        mask = ~cls.is_in(__x)
        return cls._values[mask].to_list()

    def intersection(cls: type[EnumMember], __x: Hashable | Iterable[Hashable]) -> list[EnumMember]:
        mask = cls.is_in(__x)
        return cls._values[mask].to_list()

    def map(cls, __x: Iterable[Hashable]) -> Mapping[str, EnumT]:
        return {x: cls.__call__(x) for x in map(str, __x)}


# =====================================================================================================================


EnumT_co = TypeVar("EnumT_co", bound=Any, covariant=True)


class EnumMetaBase(enum.EnumMeta, Generic[EnumT_co]):
    __iter__: Callable[..., Iterator[Self]]  # type: ignore

    @overload  # type: ignore
    def __call__(cls: type[EnumT], __item: Any) -> EnumT:
        ...

    @overload
    def __call__(cls: type[EnumT], __item: Iterable[Any], *args: Any) -> list[EnumT]:
        ...

    def __call__(cls: type[EnumT], __item: Iterable[Any], *args: Any) -> EnumT | list[EnumT]:
        if isinstance(__item, cls.__mro__[:-1]) and not args:
            return super().__call__(__item)

        return list(squish_map(super().__call__, __item, *args))

    @property
    @abc.abstractmethod
    def names(cls) -> pd.Series[str]:
        raise NotImplementedError

    def is_in(cls, x: Iterable[Any]) -> pd.Series[bool]:
        if isinstance(x, str):
            x = [x]
        return cls.names.isin(x).groupby(level=0).any()

    def difference(cls, __x: Iterable[Any]) -> set[EnumT_co]:
        mask = cls.is_in(__x)
        return {cls[v] for v in mask[~mask].index}

    def intersection(cls, __x: Iterable[Any]) -> set[Hashable]:
        mask = cls.is_in(__x)
        return {cls[v] for v in mask[mask].index}

    def map(cls, dims: Iterable[Any]):
        return {dim: cls.__call__(dim) for dim in map(str, dims)}


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.value!r}]"

    @classmethod
    def _missing_(cls, key: Any) -> Self:
        if (x := cls._member_map_.get(str(key).upper(), None)) is not None:
            return x  # type: ignore
        raise ValueError(f"{key!r} is not a valid {cls.__name__}.")


# =====================================================================================================================
class Data(Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> Iterable[tuple[str, T]]:
        ...

    def to_dict(self) -> dict[str, T]:
        return dict(self.data)

    def __repr__(self) -> str:
        data = indent_kv(*self.data)
        return "\n".join([f"{self.__class__.__name__}:"] + data)


class DataMapping(Mapping[K, T], Data[T]):
    def __init__(self, data: Mapping[K, T]) -> None:
        super().__init__()
        self._data: Final[Mapping[K, T]] = data

    @property
    def data(self) -> Iterable[tuple[K, T]]:
        yield from self.items()

    def __getitem__(self, key: K) -> T:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class DataWorker(Mapping[K, T]):
    __slots__ = ("_indices", "config")

    def __init__(self, *, indices: Iterable[K], **config: Any) -> None:
        super().__init__()
        self._indices: Final[list[K]] = list(indices)
        self.config: Final[DictStrAny] = config

    @property
    def indices(self) -> tuple[K, ...]:
        return tuple(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[K]:
        return iter(self._indices)

    def __repr__(self) -> str:
        x = self._indices
        return f"{self.__class__.__name__}(indices=[{x[0]} ... {x[-1]}])"

    def split(self, frac: float = 0.8) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self._indices[:n], self._indices[n:]
        return cls(indices=left, **self.config), cls(indices=right, **self.config)

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self._indices)
        return self

    @abc.abstractmethod
    def __getitem__(self, key: K) -> T:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...


class ABCDataConsumer(Generic[K, T], IterableDataset[T]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, worker: DataWorker[K, T], *, maxsize: int = 0, timeout: float | None = None) -> None:
        super().__init__()
        self.thread: Final = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue: Final = queue.Queue[T](maxsize=maxsize)
        self.worker: Final = worker
        self.timeout: Final = timeout

    def _target(self) -> None:
        for index in self.worker.keys():
            self.queue.put(self.worker[index], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[T]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.worker.start()
        self.thread.start()
        return self


class ABCArray(Sequence[T], Generic[S, T], abc.ABC):
    ...


class Array(ABCArray[Concatenate[P], T]):
    def __init__(self, data: NestedSequence[T] | NDArray[T]) -> None:
        super().__init__()
        self._data = np.asanyarray(data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:\n{np.array2string(self._data)}"

    def __array__(self) -> NDArray[T]:
        return self._data

    def __getitem__(self, idx: N | tuple[N, ...] | Array[..., np.bool_]) -> Array[..., T]:
        return Array(self._data[idx])

    @property
    def size(self) -> int:
        return self._data.size

    def is_in(self, x: ArrayLike) -> Array[P, np.bool_]:
        return Array(np.isin(self._data, x))  # type: ignore

    def to_numpy(self) -> NDArray[T]:
        return self._data

    def item(self) -> T:
        return self._data.item()
