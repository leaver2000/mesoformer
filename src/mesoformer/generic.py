"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import dataclasses
import enum
import os
import queue
import random
import textwrap
import threading
import types
from typing import overload

import pandas as pd
import pyproj
from torch.utils.data import IterableDataset

from .typing import (
    Any,
    DictStrAny,
    EnumT,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    ParamSpec,
    Self,
    TypeVar,
)
from .utils import find, indent_kv, load_toml, nested_proxy, squish_map

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "metadata.toml")

# =====================================================================================================================
# - Type Variables
P = ParamSpec("P")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T", bound=Any)
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)


# =====================================================================================================================
# - Enum objects and metadata
# =====================================================================================================================

_module2title = types.MappingProxyType(
    {
        f"{__package__}.{module}": title
        for module, title in [
            ("datasets.urma.constants", "URMA 2.5km CONUS"),
            ("datasets.era5.constants", "ERA5 0.25 degree GLOBAL"),
        ]
    }
)


# def get_metadata(value: str):
#     return DatasetMetadata.from_title(_module2title[find(lambda name: value in name, _module2title.keys())])


@dataclasses.dataclass(frozen=True, repr=False)
class DatasetMetadata:
    title: str
    institution: str
    source: str
    history: str
    comment: str
    coordinates: list[dict[str, Any]]
    variables: types.MappingProxyType[str, types.MappingProxyType[str, Any]]
    crs: pyproj.CRS

    @classmethod
    def from_title(cls, title: str) -> DatasetMetadata:
        title = title.upper()
        datasets = load_toml(CONFIG_FILE)["datasets"]  # type: list[dict[str, Any]]
        md = find(lambda x: x["title"] == title, datasets)
        crs = pyproj.CRS.from_cf(md.pop("crs"))
        variables = nested_proxy({dvar["standard_name"]: dvar for dvar in md.pop("variables")})

        return cls(**md, variables=variables, crs=crs)

    # @classmethod
    # def from_alias(cls, value: str) -> DatasetMetadata:
    #     title = _module2title[find(lambda name: value in name, _module2title.keys())]
    #     return cls.from_title(title)

    def __repr__(self) -> str:
        content = indent_kv(
            ("title", self.title),
            ("institution", self.institution),
            ("source", self.source),
            ("history", self.history),
            ("comment", self.comment),
            ("coordinates", self.coordinates),
            ("crs", textwrap.indent(repr(self.crs), "  ").strip()),
        )
        return "\n".join([f"{self.__class__.__name__}:"] + content)

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "short_name",
            "standard_name",
            "long_name",
            "coordinates",
            "type_of_level",
            "levels",
            "description",
            "units",
        ]
        return pd.DataFrame(list(self.variables.values()))[columns]


class CFDatasetEnumMeta(enum.EnumMeta):
    _metadata_: DatasetMetadata

    def __new__(cls, name: str, bases: tuple[Any, ...], kwargs) -> Self:
        obj = super().__new__(cls, name, bases, kwargs)
        if title := _module2title.get(kwargs["__module__"]):
            obj._metadata_ = DatasetMetadata.from_title(title)

        return obj

    @overload
    def __call__(cls: type[EnumT], __item: str) -> EnumT:
        ...

    @overload
    def __call__(cls: type[EnumT], __item: Iterable[Any], *args: Any) -> list[EnumT]:
        ...

    def __call__(cls: type[EnumT], __item: Iterable[Any], *args: Any) -> EnumT | list[EnumT]:
        if isinstance(__item, cls.__mro__[:-1]) and not args:
            return super().__call__(__item)

        return list(squish_map(super().__call__, __item, *args))

    @property
    def md(cls) -> DatasetMetadata:
        return cls._metadata_

    @property
    def crs(cls) -> pyproj.CRS:
        return cls._metadata_.crs

    def to_dataframe(cls):
        df = cls.md.to_dataframe()
        df.index = pd.Index(df["standard_name"].map(lambda x: cls(x).name).rename("member_name"))

        return df


class CFDatasetEnum(str, enum.Enum, metaclass=CFDatasetEnumMeta):
    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def _missing_(cls, key: Any) -> Self:
        return cls._member_map_[str(key).upper()]  # type: ignore

    @property
    def md(self) -> types.MappingProxyType[str, Any]:
        return self._metadata_.variables[self]

    @property
    def crs(self) -> pyproj.CRS:
        return self._metadata_.crs

    @property
    def short_name(self) -> str:
        return self.md["short_name"]

    @property
    def standard_name(self) -> str:
        return self.md["standard_name"]

    @property
    def long_name(self) -> str:
        return self.md["long_name"]

    @property
    def units(self) -> str:
        return self.md["units"]

    @property
    def type_of_level(self) -> str:
        return self.md["type_of_level"]

    @property
    def level(self) -> int:
        return self.md["level"]

    @property
    def description(self) -> str:
        return self.md["description"]


# =====================================================================================================================
# - Abstract Classes
# =====================================================================================================================
class DataMapping(Mapping[K, V], abc.ABC):
    def __init__(self, data: Mapping[K, V]) -> None:
        super().__init__()
        self._data: Final[Mapping[K, V]] = data

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[K, V]:
        return dict(self)


class DataWorker(Mapping[K, V], abc.ABC):
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
    def __getitem__(self, key: K) -> V:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...


class DataConsumer(Generic[K, V], IterableDataset[V], abc.ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, worker: DataWorker[K, V], *, maxsize: int = 0, timeout: float | None = None) -> None:
        super().__init__()
        self.thread: Final = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue: Final = queue.Queue[V](maxsize=maxsize)
        self.worker: Final = worker
        self.timeout: Final = timeout

    def _target(self) -> None:
        for index in self.worker.keys():
            self.queue.put(self.worker[index], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[V]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.worker.start()
        self.thread.start()
        return self
