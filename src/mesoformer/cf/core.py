from __future__ import annotations

import textwrap
from ..typing import Any, Final, Hashable, Iterable, Iterator, Literal, Mapping, overload, Callable, TypeVar
import types
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray


T = TypeVar("T")
_ON_DISK = [
    {
        "type": "coordinate",
        "short_name": "lon",
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "dims": ("y", "x"),
    },
    {
        "type": "coordinate",
        "short_name": "lat",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "dims": ("y", "x"),
    },
    {
        "type": "coordinate",
        "short_name": "time",
        "standard_name": "time",
        "long_name": "time",
        "units": "seconds since 1970-01-01 00:00:00",
        "dims": ("t",),
    },
    {
        "type": "coordinate",
        "short_name": "level",
        "long_name": "vertical",
        "standard_name": "vertical",
        "units": "m",
        "dims": ("z",),
    },
] + [
    # dims
    {
        "type": "dimension",
        "short_name": "x",
        "long_name": "projection_x_coordinate",
        "standard_name": "x",
        "units": "m",
        "dims": ("x",),
    },
    {
        "type": "dimension",
        "short_name": "y",
        "long_name": "projection_y_coordinate",
        "standard_name": "y",
        "units": "m",
        "dims": ("y",),
    },
    # variables
    {
        "type": "variable",
        "short_name": "ceil",
        "long_name": "cloud ceiling",
        "standard_name": "ceiling",
        "units": "m",
        "dims": ("t", "y", "x"),
    },
]


AggregationGroups = Literal["names", "types", "units", "dims"]
Testable = Hashable | list[Hashable]


def hash_list(x: Testable) -> list[Hashable]:
    return [x] if isinstance(x, Hashable) else x


class StandardName(str):
    __slots__ = ("type", "short_name", "long_name", "units", "dims")

    type: Final[str]  # type: ignore
    short_name: Final[str]  # type: ignore
    long_name: Final[str]  # type: ignore
    units: Final[str]  # type: ignore
    dims: Final[tuple[str, ...]]  # type: ignore

    def __new__(cls, **attrs: Any) -> StandardName:
        obj = str.__new__(cls, attrs.pop("standard_name"))
        for k, v in attrs.items():
            if k not in cls.__slots__:
                raise TypeError(f"Invalid attribute: {k}")
            object.__setattr__(obj, k, v)
        return obj

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise TypeError("Cannot set attribute")

    @property
    def name(self) -> str:
        return self.__str__()

    @property
    def standard_name(self) -> str:
        return self.name

    @property
    def attrs(self) -> dict[str, str]:
        return self.to_dict()

    def to_dict(self) -> dict[str, str]:
        return {"standard_name": self.__str__()} | {item: self.__getattribute__(item) for item in self.__slots__}

    def to_list(self) -> list[str]:
        return list(self.to_dict().values())

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), dtype="unicode")

    def __repr__(self) -> str:
        attrs = "\n".join(f"{item}: {self.__getattribute__(item)!r}" for item in self.__slots__)
        attrs = textwrap.indent(attrs, "  ")
        return f"{self.__class__.__name__}['{self}']\n{attrs}"

    # def get_coords(self, values:np.ndarray) -> dict[str, Any]:


class _CFConventions(Mapping[str, StandardName]):
    __slots__ = ("df", "data")
    _standard_name_slots = StandardName.__slots__

    col = types.MappingProxyType(
        {
            "names": ["short_name", "long_name", "standard_name"],
            "types": ["type"],
            "units": ["units"],
            "dims": ["dims"],
        }
    )

    def __init__(self, records: Iterable[Mapping[str, str]]) -> None:
        self.df: Final = pd.DataFrame(list(records))
        self.df["dims"] = self.df["dims"].apply(tuple)
        self.data: Final = [StandardName(**attrs) for attrs in self.df.to_dict(orient="records")]  # type: ignore

    @property
    def names(self):
        return self.df[self.col["names"]]

    @property
    def types(self):
        return self.df[self.col["type"]]

    @property
    def units(self):
        return self.df[self.col["units"]]

    @property
    def dims(self):
        return self.df[self.col["dims"]]

    @overload
    def __getitem__(self, name: str) -> StandardName:
        ...

    @overload
    def __getitem__(self, name: list[str]) -> list[StandardName]:
        ...

    def __getitem__(self, name: str | list[str]) -> StandardName | list[StandardName]:
        idx = self.nonzero("names", name)  # type: ignore

        if not isinstance(name, str):
            return [self.data[i] for i in idx]
        elif len(idx) != 1:
            raise KeyError(name)

        return self.data[idx[0]]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.map_names(str))

    def __repr__(self) -> str:
        return "\n".join(self.map_names(repr))

    def map_names(self, func: Callable[[StandardName], T]) -> map[T]:
        return map(func, self.data)

    def get_group(self, by: AggregationGroups) -> pd.DataFrame:
        if by not in self.col:
            raise ValueError(f"Invalid group: {by}")
        return self.df[self.col[by]]

    def get_mapping(self, name: Iterable[str]) -> dict[str, StandardName]:
        return {x: self[x] for x in name}

    def isin(self, by: AggregationGroups, test: Testable) -> NDArray[np.bool_]:
        return self.get_group(by).isin(hash_list(test)).to_numpy().any(axis=1)

    def nonzero(self, by: AggregationGroups, test: Testable) -> NDArray[np.int_]:
        (indices,) = self.isin(by, test).nonzero()
        return indices

    def agg(self, by: AggregationGroups, key: Testable) -> list[StandardName]:
        return [self.data[idx] for idx in self.nonzero(by, key)]


# - conventions
conventions: Final = _CFConventions(_ON_DISK)

# - coords
time: Final = conventions["time"]
level: Final = conventions["level"]
lon: Final = conventions["longitude"]
lat: Final = conventions["latitude"]

# - dims
x: Final = conventions["x"]
y: Final = conventions["y"]


def is_coordinate(s: StandardName) -> bool:
    return s.type == "coordinate"


def is_dimension(s: StandardName) -> bool:
    return s.type == "dimension"


def is_variable(s: StandardName) -> bool:
    return s.type == "variable"


def create_data_array(standard: StandardName, values: np.ndarray | xr.DataArray) -> xr.DataArray:
    if not is_variable(standard):
        coords = {standard.name: (standard.dims, values)}
    else:
        raise NotImplementedError
        coords = {name.name: (name.dims, values.to_numpy()) for dim in conventions[list(name.dims)]}

    return xr.DataArray(name=standard.name, dims=standard.dims, coords=coords, attrs=standard.attrs)


CoordinateMap = Iterable[tuple[StandardName, NDArray[np.float_]]]


def create_coordinates(data_coords: CoordinateMap) -> dict[str, xr.DataArray]:
    return {
        coord.name: xr.DataArray(data, name=coord.name, dims=coord.dims, attrs=coord.attrs)
        for coord, data in data_coords
    }


# def test_standard_name(
#     standard_name: str,
#     short_name: str,
#     long_name: str,
#     type: str,
#     units: str,
#     dims: str,
# ):
#     assert conventions[standard_name] == conventions[short_name] == conventions[long_name]
#     value = conventions[standard_name]
#     assert isinstance(value, StandardName)
#     assert value == standard_name
#     assert value.short_name == short_name
#     assert value.long_name == long_name
#     for x in conventions.agg("types", type):
#         assert x.type == type

#     for x in conventions.agg("units", units):
#         assert x.units == units

#     for x in conventions.agg("dims", dims):
#         assert x.dims == dims


# def test():
#     for record in _ON_DISK:
#         test_standard_name(**record)


# if __name__ == "__main__":
#     test()
