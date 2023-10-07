from __future__ import annotations

__all__ = [
    # - ERA5
    "ERA5Enum",
    "ERA5_VARS",
    "GEOPOTENTIAL",
    "SPECIFIC_HUMIDITY",
    "TEMPERATURE",
    "U_COMPONENT_OF_WIND",
    "V_COMPONENT_OF_WIND",
    "VERTICAL_VELOCITY",
    # - URMA
    "URMAEnum",
    "URMA_VARS",
    "CEILING",
    "DEWPOINT_TEMPERATURE_2M",
    "OROGRAPHY",
    "SPECIFIC_HUMIDITY_2M",
    "SURFACE_PRESSURE",
    "TEMPERATURE_2M",
    "TOTAL_CLOUD_COVER",
    "URMAEnum",
    "U_WIND_COMPONENT_10M",
    "VISIBILITY",
    "V_WIND_COMPONENT_10M",
    "WIND_DIRECTION_10M",
    "WIND_SPEED_10M",
    "WIND_SPEED_GUST",
]


import abc
import dataclasses
import enum
import textwrap
import types

from typing import Iterable, Literal, TypeAlias, TypeVar, Mapping, Sequence, Any, Hashable

import pandas as pd
import pyproj

from ..config import get_dataset
from ..generic import Data, EnumMetaBase, StrEnum
from ..typing import DictStrAny, Self
from ..utils import nested_proxy


# =====================================================================================================================
#  Coordinate and dimension conventions
# =====================================================================================================================
class Convention(str):
    __slots__ = ("axis", "standard_names")
    axis: tuple[str, ...]
    standard_names: tuple[str, ...]

    def __new__(
        cls,
        name: DictStrAny | Convention | str | Hashable,
        *,
        axis: tuple[Convention, ...] | None = None,
        standard_names: tuple[str, ...] | None = None,
        units: str | None = None,
    ) -> Convention:
        if isinstance(name, dict):
            standard_names = name.get("standard_name", [name["name"]])
            axis = name.get("axis", [name["name"]])
            name = name["name"]
        assert isinstance(name, str)
        obj = super().__new__(cls, name)
        obj.axis = tuple(sorted(set(axis or [obj])))
        obj.standard_names = tuple(sorted(set(standard_names or [name])))
        return obj


class ConventionMeta(EnumMetaBase["ConventionEnum"]):
    standard_names: str

    @property
    def names(cls):
        names = [member.name for member in cls]
        return pd.DataFrame(
            [list({member.name, str(member.value)}.union(member.standard_names)) for member in cls], index=names
        ).stack()


class ConventionEnum(Convention, enum.Enum, metaclass=ConventionMeta):  # type: ignore
    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    @classmethod
    def _missing_(cls, name: Any):
        for member in cls:
            if member.casefold() == name.casefold():
                return member
            if name in member.standard_names:
                return member


class Dimensions(ConventionEnum):
    T = {
        "name": "T",
        "standard_name": ["time"],
    }
    """
    Variables representing time must always explicitly include the units attribute; there is no default
    value. The units attribute takes a string value formatted as per the recommendations in the
    [UDUNITS] package. The following excerpt from the UDUNITS documentation explains the time unit
    encoding by example:
    """
    Z = {
        "name": "Z",
        "standard_name": ["level", "height", "altitude"],
    }
    Y = {
        "name": "Y",
        "standard_name": ["latitude", "grid_latitude"],
    }
    X = {
        "name": "X",
        "standard_name": ["longitude", "grid_longitude"],
    }


LiteralOrder: TypeAlias = tuple[
    Literal[Dimensions.T],
    Literal[Dimensions.Z],
    Literal[Dimensions.Y],
    Literal[Dimensions.X],
]
DIMENSIONS = T, Z, Y, X = (
    Dimensions.T,
    Dimensions.Z,
    Dimensions.Y,
    Dimensions.X,
)


class Coordinates(ConventionEnum):
    time = {
        "name": "time",
        "axis": (T,),
        "standard_name": ["time"],
    }
    vertical = {
        "name": "vertical",
        "axis": (Z,),
        "standard_name": ["level", "height", "altitude"],
    }
    latitude = {
        "name": "latitude",
        "axis": (Y, X),
        "standard_name": ["latitude", "grid_latitude"],
    }
    longitude = {
        "name": "longitude",
        "axis": (Y, X),
        "standard_name": ["longitude", "grid_longitude"],
    }


COORDINATES = TIME, LVL, LAT, LON = (
    Coordinates.time,
    Coordinates.vertical,
    Coordinates.latitude,
    Coordinates.longitude,
)


# =====================================================================================================================
#  - Dataset metadata
# =====================================================================================================================


_T = TypeVar("_T")
SeriesOrType: TypeAlias = "pd.Series[_T] | _T"  # type: ignore


class MetaVarMixin(Data[Any], abc.ABC):
    @property
    @abc.abstractmethod
    def dvars(self) -> Mapping[str, Any]:
        ...

    @property
    def short_name(self) -> SeriesOrType[str]:
        return self.dvars["short_name"]

    @property
    def standard_name(self) -> str:
        return self.dvars["standard_name"]

    @property
    def long_name(self) -> str:
        return self.dvars["long_name"]

    @property
    def units(self) -> str:
        return self.dvars["units"]

    @property
    def type_of_level(self) -> str:
        return self.dvars["type_of_level"]

    @property
    def level(self) -> int:
        return self.dvars["level"]

    @property
    def description(self) -> str:
        return self.dvars["description"]

    @property
    def coords(self) -> list[str]:
        return self.dvars["coordinates"]


@dataclasses.dataclass(frozen=True, repr=False)
class DatasetMetadata(MetaVarMixin):
    title: str
    institution: str
    source: str
    history: str
    comment: str
    coordinates: list[dict[str, Any]]
    variables: types.MappingProxyType[str, types.MappingProxyType[str, Any]]
    crs: pyproj.CRS

    def __post_init__(self):
        object.__setattr__(self, "_dataframe", pd.DataFrame(list(self.variables.values())))

    @classmethod
    def from_title(cls, title: str) -> DatasetMetadata:
        md = get_dataset(title)
        crs = pyproj.CRS.from_cf(md.pop("crs"))
        variables = nested_proxy({dvar["standard_name"]: dvar for dvar in md.pop("variables")})
        return cls(**md, variables=variables, crs=crs)

    @property
    def metadata(self) -> DatasetMetadata:
        return self

    @property
    def dvars(self) -> pd.DataFrame:
        return self._dataframe  # type: ignore

    def _set_index(self, enum_cls) -> None:
        self.dvars.index = pd.Index(self.dvars["standard_name"].map(lambda x: enum_cls(x).name).rename("member_name"))

    @property
    def data(self) -> Iterable[tuple[str, Any]]:
        yield from [
            ("title", self.title),
            ("institution", self.institution),
            ("source", self.source),
            ("history", self.history),
            ("comment", self.comment),
            ("coordinates", self.coordinates),
            ("crs", "\n" + textwrap.indent(repr(self.crs), "  ").strip()),
        ]

    def to_dataframe(self) -> pd.DataFrame:
        return self.dvars.copy()


class MetadataMixin(MetaVarMixin, abc.ABC):
    @property
    @abc.abstractmethod
    def metadata(self) -> DatasetMetadata:
        ...

    @property
    def md(cls) -> DatasetMetadata:
        return cls.metadata

    @property
    def crs(cls) -> pyproj.CRS:
        return cls.md.crs

    @property
    def data(self):
        return self.md.data

    def get_coords(self) -> list[Hashable]:
        return [coord["standard_name"] for coord in self.md.coordinates]


class CFDatasetEnumMeta(MetadataMixin, EnumMetaBase[StrEnum]):
    """
    A metaclass for creating Enum classes that have associated metadata.

    Attributes:
        _metadata_: An instance of DatasetMetadata that contains metadata for the Enum class.
    """

    _metadata_: DatasetMetadata

    def __new__(cls, name: str, bases: tuple[Any, ...], kwargs, title: str | None = None) -> Self:  # pyright: ignore
        obj = super().__new__(cls, name, bases, kwargs)
        if title is not None:
            obj._metadata_ = md = DatasetMetadata.from_title(title)
            md._set_index(obj)

        return obj

    @property
    def metadata(cls) -> DatasetMetadata:
        """
        Returns the metadata associated with the Enum class.

        Returns:
            An instance of DatasetMetadata that contains metadata for the Enum class.
        """
        return cls._metadata_

    @property
    def dvars(cls) -> pd.DataFrame:
        return cls.metadata.dvars

    @property
    def names(self) -> pd.Series[str]:
        return self.dvars[["long_name", "short_name", "standard_name"]].stack()  # type: ignore


class CFDatasetEnum(StrEnum, metaclass=CFDatasetEnumMeta):
    @property
    def dvars(self) -> types.MappingProxyType[str, Any]:
        return self._metadata_.variables[self]  # type: ignore


# =====================================================================================================================
class ERA5Enum(CFDatasetEnum, title="ERA5"):
    r"""
    | member_name   | short_name   | standard_name       | long_name           | type_of_level   | units      |
    |:--------------|:-------------|:--------------------|:--------------------|:----------------|:-----------|
    | Z             | z            | geopotential        | Geopotential        | isobaricInhPa   | m**2 s**-2 |
    | Q             | q            | specific_humidity   | Specific humidity   | isobaricInhPa   | kg kg**-1  |
    | T             | t            | temperature         | Temperature         | isobaricInhPa   | K          |
    | U             | u            | u_component_of_wind | U component of wind | isobaricInhPa   | m s**-1    |
    | V             | v            | v_component_of_wind | V component of wind | isobaricInhPa   | m s**-1    |
    | W             | w            | vertical_velocity   | Vertical velocity   | isobaricInhPa   | Pa s**-1   |
    """

    Z = "geopotential"
    Q = "specific_humidity"
    T = "temperature"
    U = "u_component_of_wind"
    V = "v_component_of_wind"
    W = "vertical_velocity"


ERA5_VARS = (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    VERTICAL_VELOCITY,
) = (
    ERA5Enum.Z,
    ERA5Enum.Q,
    ERA5Enum.T,
    ERA5Enum.U,
    ERA5Enum.V,
    ERA5Enum.W,
)


class URMAEnum(CFDatasetEnum, title="URMA"):
    """
    | member_name   | short_name   | standard_name           | long_name                    | type_of_level         | units       |
    |:--------------|:-------------|:------------------------|:-----------------------------|:----------------------|:------------|
    | CEIL          | ceil         | ceiling                 | cloud ceiling                | cloudCeiling          | m           |
    | D2M           | d2m          | dewpoint_temperature_2m | 2 meter dewpoint temperature | heightAboveGround     | K           |
    | SH2           | sh2          | specific_humidity_2m    | 2 meter specific humidity    | heightAboveGround     | kg kg**-1   |
    | SP            | sp           | surface_pressure        | surface pressure             | surface               | Pa          |
    | T2M           | t2m          | temperature_2m          | 2 meter temperature          | heightAboveGround     | K           |
    | TCC           | tcc          | total_cloud_cover       | total cloud cover            | atmosphereSingleLayer | %           |
    | U10           | u10          | u_wind_component_10m    | 10 meter u wind component    | heightAboveGround     | m s**-1     |
    | V10           | v10          | v_wind_component_10m    | 10 meter v wind component    | heightAboveGround     | m s**-1     |
    | VIS           | vis          | visibility              | visibility                   | surface               | m           |
    | WDIR10        | wdir10       | wind_direction_10m      | 10 meter wind direction      | heightAboveGround     | Degree true |
    | SI10          | si10         | wind_speed_10m          | 10 meter wind speed          | heightAboveGround     | m s**-1     |
    | GUST          | gust         | wind_speed_gust         | wind speed gust              | heightAboveGround     | m s**-1     |
    | OROG          | orog         | orography               | surface orography            | surface               | m           |
    """

    TCC = "total_cloud_cover"
    CEIL = "ceiling"
    U10 = "u_wind_component_10m"
    V10 = "v_wind_component_10m"
    SI10 = "wind_speed_10m"
    GUST = "wind_speed_gust"
    WDIR10 = "wind_direction_10m"
    T2M = "temperature_2m"
    D2M = "dewpoint_temperature_2m"
    SH2 = "specific_humidity_2m"
    SP = "surface_pressure"
    VIS = "visibility"
    OROG = "orography"


URMA_VARS = (
    TOTAL_CLOUD_COVER,
    CEILING,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    WIND_SPEED_10M,
    WIND_SPEED_GUST,
    WIND_DIRECTION_10M,
    TEMPERATURE_2M,
    DEWPOINT_TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    VISIBILITY,
    OROGRAPHY,
) = (
    URMAEnum.TCC,
    URMAEnum.CEIL,
    URMAEnum.U10,
    URMAEnum.V10,
    URMAEnum.SI10,
    URMAEnum.GUST,
    URMAEnum.WDIR10,
    URMAEnum.T2M,
    URMAEnum.D2M,
    URMAEnum.SH2,
    URMAEnum.SP,
    URMAEnum.VIS,
    URMAEnum.OROG,
)


_ENUM_REGISTRY: Sequence[type[CFDatasetEnum]] = (ERA5Enum, URMAEnum)


def register_enum(enum_cls: type[CFDatasetEnum]) -> None:
    global _ENUM_REGISTRY
    _ENUM_REGISTRY = tuple(_ENUM_REGISTRY) + (enum_cls,)


def find_enums(vrbs: str | list[str]) -> list[CFDatasetEnum]:
    return [member for enum_ in _ENUM_REGISTRY for member in enum_.intersection(vrbs)]  # type: ignore
