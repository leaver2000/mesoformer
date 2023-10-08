from __future__ import annotations

import enum
import pandas as pd
import abc
from typing import Generic, TypeVar, Any, Callable, Iterable, Iterator, Hashable, Union, overload, TypeAlias, Mapping


EnumT = TypeVar("EnumT")
Aliases = Mapping[str, list[str]]


class GenericEnumMeta(enum.EnumMeta, Generic[EnumT]):
    def __class_getitem__(cls, __x: Any) -> TypeAlias:
        return cls


class SeriesMeta(GenericEnumMeta[EnumT]):
    __iter__: Callable[..., Iterator[EnumT]]  # type: ignore[assignment]
    _member_map_: pd.DataFrame

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        cls_dict: enum._EnumDict,
        aliases: Aliases | None = None,
    ):
        obj = super().__new__(cls, name, bases, cls_dict)
        if aliases is None:
            aliases = obj.get_aliases()
        obj._member_map_ = pd.DataFrame.from_dict(
            {k: [v, *aliases.get(k, [])] for k, v in obj._member_map_.items()}, orient="index"
        ).astype("string")
        return obj

    @property
    def values(self) -> pd.Series[EnumT]:
        return self._member_map_.iloc[:, 0]

    @overload  # type: ignore
    def __getitem__(self, names: str) -> EnumT:
        ...

    @overload
    def __getitem__(self, names: list[str]) -> list[EnumT]:
        ...

    def __getitem__(self, names: str | list[str]) -> EnumT | list[EnumT]:
        x = self._member_map_.loc[names, 0]
        if isinstance(x, pd.Series):
            return x.to_list()
        return x

    @abc.abstractmethod
    def get_aliases(cls) -> Mapping[str, list[str]]:
        ...

    def to_series(cls) -> pd.Series[str]:
        return cls._member_map_

    def is_in(cls, __x: Iterable[Any]) -> pd.Series[bool]:
        if isinstance(__x, str):
            __x = [__x]
        return cls.to_series().isin(__x).groupby(level=0).any()

    def difference(cls, __x: Iterable[Any]) -> set[EnumT]:
        mask = cls.is_in(__x)
        return {cls[v] for v in mask[~mask].index}

    def intersection(cls, __x: Iterable[Any]) -> set[EnumT]:
        mask = cls.is_in(__x)
        return {cls[v] for v in mask[mask].index}

    def map(cls, __x: Iterable[Any]) -> Mapping[str, EnumT]:
        return {x: cls.__call__(x) for x in map(str, __x)}


class MyEnum(
    str,
    enum.Enum,
    metaclass=SeriesMeta["MyEnum"],  # type: ignore[misc]
    aliases={
        "A": ["alpha", "apple"],
        "B": ["bravo"],
        "C": ["charlie"],
    },
):
    A = "a"
    B = "b"
    C = "c"


assert MyEnum["A"] == MyEnum.A
assert MyEnum("b") == MyEnum.B
