"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

__all__ = [
    "auto_field",
    "TableEnum",
    "TableEnumMeta",
]
import collections
import enum
import types
from typing import MutableMapping, NamedTuple, TypeAlias

import pandas as pd

from .typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Hashable,
    HashableT,
    Iterable,
    Iterator,
    ListLike,
    Mapping,
    MaskType,
    MutableMapping,
    NamedTuple,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)
from .utils import is_scalar

_ENUM_ALIASES = "_aliases_"
_ENUM_DICT_RESERVED_KEYS = (
    "__doc__",
    "__module__",
    "__qualname__",
    #
    "_order_",
    "_create_pseudo_member_",
    "_generate_next_value_",
    "_missing_",
    "_ignore_",
)

MemberMetadata: TypeAlias = MutableMapping[str, Any]


_Items: TypeAlias = list[Hashable] | pd.Index | pd.Series | slice | MaskType
_T = TypeVar("_T")


EnumT = TypeVar("EnumT", bound="_EnumSelf")
if TYPE_CHECKING:

    class _EnumSelf(Protocol):
        _data: ClassVar[pd.DataFrame]
        _values: ClassVar[pd.Series[Any]]
        _names: ClassVar[pd.Index[str]]

        @classmethod
        def __getitem__(cls, __names: ListLike[bool] | ListLike[Hashable] | bool | Hashable) -> Self | list[Self]:
            ...

        @classmethod
        def __iter__(cls) -> Iterator[Self]:
            ...

        @classmethod
        def is_in(cls, __x: Hashable | Iterable[Hashable]) -> pd.Series[bool]:
            ...

        @classmethod
        def intersection(cls, __x: Hashable | Iterable[Hashable]) -> list[Self]:
            ...

        @classmethod
        def difference(cls, __x: Hashable | Iterable[Hashable]) -> list[Self]:
            ...

        @classmethod
        def remap(cls, __x: Hashable | Iterable[Hashable]) -> Mapping[Hashable, Self]:
            ...

    _S1 = TypeVar("_S1", bound=Any)
    Series: TypeAlias = pd.Series[_S1]
else:
    Series: TypeAlias = pd.Series | Sequence[_T]


class _Field(NamedTuple):
    value: Any
    metadata: Mapping[str, Any]


def auto_field(value: _T | Any = None, *, aliases: list[_T] | None = None, **metadata: Any) -> Any:
    if value is None:
        value = enum.auto()
    if _ENUM_ALIASES in metadata and aliases is None:
        assert isinstance(metadata[_ENUM_ALIASES], list)
    elif _ENUM_ALIASES in metadata and aliases is not None:
        raise ValueError("Field metadata contains aliases and aliases were passed as an argument.")
    else:
        metadata[_ENUM_ALIASES] = aliases or []

    return _Field(value, metadata)


def _unpack_info(old: enum._EnumDict) -> tuple[enum._EnumDict, dict[str, Any]]:
    """Unpacks the enum_dict into a new dict and a metadata dict."""
    new = enum._EnumDict()
    new._cls_name = old._cls_name  # type: ignore
    meta = {}  # type: dict[str, Any]
    for key, value in old.items():
        if isinstance(value, _Field):
            new[key], meta[key] = value
        else:
            new[key] = value
            if key not in _ENUM_DICT_RESERVED_KEYS:
                meta[key] = {}

    return new, meta


def _repack_info(
    name: str, member_map: Mapping[str, enum.Enum], metadata: dict[str, dict[str, Any]], class_metadata: dict[str, Any]
) -> types.MappingProxyType[str, Any]:
    data = {k: [v, *set(metadata[k].pop(_ENUM_ALIASES, []))] for k, v in member_map.items()}
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "name"

    if df.duplicated().any():
        raise ValueError(f"Duplicate values found in {name}._member_map_.")

    member_metadata = types.MappingProxyType(collections.defaultdict(dict, metadata))
    return types.MappingProxyType(
        {
            "name": name,
            "data": df,
            "class_metadata": class_metadata,
            "member_metadata": member_metadata,
        }
    )


class Descriptor:
    __getitem__: Any
    _data: collections.defaultdict[int, Mapping]  # [int, types.MappingProxyType[str, Any]]

    def __init__(self) -> None:
        self._data = collections.defaultdict(dict, {})

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._data[hash(instance)]

    def __set__(self, instance, value):
        if instance is None:
            raise TypeError("Cannot set a class attribute.")
        self._data[hash(instance)] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"


class TableEnumMeta(enum.EnumMeta):
    __metadata__ = Descriptor()

    def describe(cls) -> None:
        for v in cls.__class__.__metadata__._data.values():  # type: ignore
            print(
                (
                    f"""\
{v['name']}:
alias:
{v['data']}
member_metadata:
{dict(v['member_metadata'])}
class_metadata:
{dict(v['class_metadata'])}\n"""
                )
            )

    def __new__(
        cls,
        name: str,
        bases: tuple[Any, ...],
        cls_dict: enum._EnumDict,
        **kwargs: Any,
    ):
        cls_dict, member_metadata = _unpack_info(cls_dict)
        obj = super().__new__(cls, name, bases, cls_dict)
        if obj._member_names_:
            obj.__metadata__ = _repack_info(name, obj._member_map_, member_metadata, kwargs)

        return obj

    # =================================================================================================================
    # - metadata properties
    @property
    def metadata(cls) -> MutableMapping[str, Any]:
        return cls.__metadata__["class_metadata"]

    @property
    def _member_metadata(cls) -> types.MappingProxyType[str, MemberMetadata]:
        return cls.__metadata__["member_metadata"]

    @property
    def _data(cls) -> pd.DataFrame:
        return cls.__metadata__["data"]

    # =================================================================================================================
    # - dataframe properties
    @property
    def _names(cls) -> pd.Index[str]:
        return cls._data.index

    @property
    def _values(cls: type[EnumT]) -> Series[EnumT]:
        return cls._data.iloc[:, 0]

    def to_frame(cls) -> pd.DataFrame:
        df = cls._data.copy()
        df.columns = [
            "value",
            *(f"a{i}" for i in range(df.columns.size - 1)),
        ]
        return df

    def to_series(cls: type[EnumT]) -> Series[EnumT]:
        return cls._values.copy()

    # =================================================================================================================

    if TYPE_CHECKING:

        @overload  # type: ignore
        def __getitem__(__cls: type[EnumT], __item: Hashable) -> EnumT:
            ...

        @overload
        def __getitem__(__cls: type[EnumT], __item: _Items) -> list[EnumT]:
            ...

        @overload
        @classmethod
        def __getitem__(cls, __item: Hashable) -> Any:
            ...

        @overload
        @classmethod
        def __getitem__(cls, __item: _Items) -> list[Any]:
            ...

        @overload
        def __getitem__(cls, __item) -> Any:
            ...

    def __getitem__(cls, __item: Hashable | _Items) -> Any:
        x = cls._values[__item]  # type: ignore
        if isinstance(x, pd.Series):
            return x.to_list()
        return x

    if TYPE_CHECKING:

        @overload  # type: ignore
        def __call__(cls: type[EnumT], __items: Hashable) -> EnumT:
            ...

        @overload  # type: ignore
        def __call__(cls: type[EnumT], __items: Iterable[Hashable]) -> list[EnumT]:
            ...

    def __call__(  # type: ignore
        cls: type[EnumT],
        __items: Iterable[Hashable] | Hashable,
    ) -> EnumT | list[EnumT]:
        """It is possible to return multiple members if the members share an alias."""
        if is_scalar(__items):
            mask = (cls._data == __items).any(axis=1)
            if (names := cls._names[mask]).size == 0:
                raise ValueError(f"{__items!r} is not a valid value for {cls.__name__}.")
            if len(names) == 1:
                return cls._values[names[0]]
            return cls._values[names].to_list()

        return cls.intersection(__items)

    def to_list(cls: type[EnumT]) -> list[EnumT]:
        return cls._values.to_list()

    def is_in(cls, __x: str | Iterable[Hashable]) -> Series[bool]:
        if isinstance(__x, str):
            __x = [__x]
        return cls._data.isin(__x).any(axis=1) | cls._names.isin(__x)

    def difference(cls: type[EnumT], __x: Hashable | Iterable[Hashable]) -> list[EnumT]:
        mask = ~cls.is_in(__x)
        return cls._values[mask].to_list()

    def intersection(cls: type[EnumT], __x: Hashable | Iterable[Hashable]) -> list[EnumT]:
        mask = cls.is_in(__x)
        return cls._values[mask].to_list()

    def remap(cls: type[EnumT], __x: Iterable[HashableT]) -> dict[HashableT, EnumT]:
        return {x: cls.__call__(x) for x in __x}


class TableEnum(enum.Enum, metaclass=TableEnumMeta):
    @property
    def aliases(self) -> list[Any]:
        return self.__class__._data.loc[self.name, 1:].dropna().to_list()  # type: ignore

    @property
    def metadata(self) -> MemberMetadata:
        return self.__class__._member_metadata[self.name]
