"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import enum
import collections

import pandas as pd
from typing import NamedTuple

from .typing import (
    Any,
    Protocol,
    TypeVar,
    ClassVar,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Self,
    TypeVar,
    overload,
    ListLike,
    TYPE_CHECKING,
)


_T = TypeVar("_T")
EnumT = TypeVar("EnumT", bound="_EnumSelf")


if TYPE_CHECKING:  # type: ignore

    class _EnumSelf(Protocol):  # type: ignore
        _names: ClassVar[pd.Index[str]]  # type: ignore
        _values: ClassVar[pd.Series]
        _table_: ClassVar[pd.DataFrame]

        @classmethod
        def __getitem__(cls, __names: ListLike[bool] | ListLike[Hashable] | bool | Hashable) -> Self | list[Self]:
            ...

        @classmethod
        def __iter__(cls) -> Iterator[Self]:
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


class _Field(NamedTuple):
    value: Any
    metadata: Mapping[str, Any]


def field(
    value: _T | None = None,
    *,
    aliases: list[_T] | None = None,
    metadata: Any = None,
) -> _T:
    if value is None:
        value = enum.auto()  # type: ignore
    metadata = metadata or {}
    if "_aliases_" in metadata and aliases is None:
        assert isinstance(metadata["_aliases_"], list)
    elif "_aliases_" in metadata and aliases is not None:
        raise ValueError("Field metadata contains aliases and aliases were passed as an argument.")
    else:
        metadata["_aliases_"] = aliases or []

    return _Field(value, metadata)  # type: ignore


def _unpack_info(enum_dict: enum._EnumDict) -> tuple[enum._EnumDict, dict[str, Any]]:
    self = enum._EnumDict()
    self._cls_name = enum_dict._cls_name  # type: ignore
    metadata = {}  # type: ignore
    for key, value in enum_dict.items():
        if isinstance(value, _Field):
            self[key], metadata[key] = value
        else:
            self[key] = value
    return self, metadata


def _repack_info(
    name: str, member_map: Mapping[str, enum.Enum], metadata: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    data = {k: [v, *set(metadata.get(k, {}).pop("_aliases_", []))] for k, v in member_map.items()}
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = name
    if df.duplicated().any():
        raise ValueError(f"Duplicate values found in {name}._member_map_.")
    return df, collections.defaultdict(dict, metadata)


class TableEnumMeta(enum.EnumMeta):
    _table_: pd.DataFrame
    _metadata_: Mapping[str, Any]

    def __new__(
        cls,
        name: str,
        bases: tuple[Any, ...],
        cls_dict: enum._EnumDict,
    ):
        cls_dict, metadata = _unpack_info(cls_dict)

        obj = super().__new__(cls, name, bases, cls_dict)
        if obj._member_names_:
            obj._table_, obj._metadata_ = _repack_info(name, obj._member_map_, metadata)

        return obj

    if TYPE_CHECKING:

        @overload  # type: ignore
        def __getitem__(__cls: type[EnumT], __item: str) -> EnumT:
            ...

        @overload
        def __getitem__(__cls: type[EnumT], __item: ListLike[bool | Hashable]) -> list[EnumT]:
            ...

        @overload
        @classmethod
        def __getitem__(cls, __item: str) -> Any:
            ...

        @overload
        @classmethod
        def __getitem__(cls, __item: ListLike[bool | Hashable]) -> list[Any]:
            ...

        @overload
        def __getitem__(cls, _) -> Any:
            ...

    def __getitem__(cls, __item):
        x = cls._values[__item]
        if isinstance(x, pd.Series):
            return x.to_list()
        return x

    @overload  # type: ignore
    def __call__(cls: type[EnumT], __items: str | Hashable) -> EnumT:
        ...

    @overload  # type: ignore
    def __call__(cls: type[EnumT], __items: Iterable[Hashable]) -> list[EnumT]:
        ...

    def __call__(
        cls: type[EnumT],
        __items: str | Iterable[Hashable] | Hashable,
    ) -> EnumT | list[EnumT]:
        if isinstance(__items, str):
            names = cls._names[(cls._table_ == __items).any(axis=1)]
            if len(names) == 1:
                return cls._values[names[0]]
            raise ValueError(f"Multiple values found for {__items}.")
        return cls.intersection(__items)

    def to_frame(cls) -> pd.DataFrame:
        return cls._table_

    def to_series(cls: type[EnumT]) -> pd.Series[EnumT]:
        return cls._values

    @property
    def _names(cls) -> pd.Index[str]:
        return cls._table_.index

    @property
    def _values(cls) -> pd.Series[EnumT]:
        return cls._table_.iloc[:, 0]

    def to_list(cls) -> list[EnumT]:
        return cls._values.to_list()

    def is_in(cls, __x: Hashable | Iterable[Hashable]) -> pd.Series[bool]:
        if isinstance(__x, str):
            __x = [__x]
        return cls.to_frame().isin(__x).any(axis=1) | cls._names.isin(__x)

    def difference(cls: type[EnumT], __x: Hashable | Iterable[Hashable]) -> list[EnumT]:
        mask = ~cls.is_in(__x)
        return cls._values[mask].to_list()

    def intersection(cls: type[EnumT], __x: Hashable | Iterable[Hashable]) -> list[EnumT]:
        mask = cls.is_in(__x)
        return cls._values[mask].to_list()

    def map(cls, __x: Iterable[Hashable]) -> Mapping[str, EnumT]:
        return {x: cls.__call__(x) for x in map(str, __x)}


class TableEnum(enum.Enum, metaclass=TableEnumMeta):
    _metadata_: ClassVar[Mapping[str, Any]]
    _table_: ClassVar[pd.DataFrame]

    @property
    def aliases(self) -> list[Any]:
        return self._table_.loc[self.name, 1:].tolist()

    @property
    def metadata(self) -> Any:
        return self._metadata_[self.name]


def main() -> None:
    class MyEnum(str, TableEnum):
        A = field(
            "a",
            aliases=["alpha"],
            metadata={"hello": "world"},
        )
        B = field("b", aliases=["beta"])
        C = field("c", aliases=["beta"])

    assert MyEnum.A == "a"
    ab = MyEnum[["A", "B"]]
    assert MyEnum["A"] == "a" == MyEnum.A == MyEnum("alpha")

    assert ab == [MyEnum.A, MyEnum.B]
    assert MyEnum.A.metadata == {"hello": "world"}
    assert MyEnum.A.aliases == ["alpha"]


if __name__ == "__main__":
    main()
