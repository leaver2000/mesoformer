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


import dataclasses
import enum
import textwrap
import types

import pandas as pd
import pyproj
from typing import Mapping, Hashable
import abc
from ..config import get_dataset
from ..typing import Any, EnumT, Iterable, Self, overload, Literal, TypeAlias
from ..utils import indent_kv, nested_proxy, squish_map
from ..generic import Data, DataMapping, EnumMetaBase, StrEnum

# from src.mesoformer.generic import StrEnum, EnumMetaBase
LiteralOrder: TypeAlias = """tuple[
    Literal[OrderedDims.TIME],
    Literal[OrderedDims.LEVEL],
    Literal[OrderedDims.LATITUDE],
    Literal[OrderedDims.LONGITUDE],
]"""


class OrderedDims(StrEnum, metaclass=EnumMetaBase):
    TIME = "t"
    LEVEL = "z"
    LATITUDE = "y"
    LONGITUDE = "x"

    @classmethod
    @property
    def order(cls) -> LiteralOrder:
        return ORDERED_DIMS


ORDERED_DIMS = T, Z, Y, X = (
    OrderedDims.TIME,
    OrderedDims.LEVEL,
    OrderedDims.LATITUDE,
    OrderedDims.LONGITUDE,
)


class MetadataMixin(Data[Any], abc.ABC):
    __ordered_dims = OrderedDims

    @property
    def dims(self) -> type[OrderedDims]:
        return self.__ordered_dims

    @property
    @abc.abstractmethod
    def metadata(self) -> DatasetMetadata:
        ...

    @property
    def md(cls) -> DatasetMetadata:
        return cls.metadata

    @property
    def crs(cls) -> pyproj.CRS:
        return cls.metadata.crs

    @property
    def data(self):
        return self.metadata.data

    def get_coords(self) -> list[Hashable]:
        return [coord["standard_name"] for coord in self.metadata.coordinates]


@dataclasses.dataclass(frozen=True, repr=False)
class DatasetMetadata(Data[Any]):
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
        md = get_dataset(title)
        crs = pyproj.CRS.from_cf(md.pop("crs"))
        variables = nested_proxy({dvar["standard_name"]: dvar for dvar in md.pop("variables")})

        return cls(**md, variables=variables, crs=crs)

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


class CFDatasetEnumMeta(MetadataMixin, EnumMetaBase):
    """
    A metaclass for creating Enum classes that have associated metadata.

    Attributes:
        _metadata_: An instance of DatasetMetadata that contains metadata for the Enum class.
    """

    _metadata_: DatasetMetadata

    def __new__(cls, name: str, bases: tuple[Any, ...], kwargs, title: str | None = None) -> Self:
        obj = super().__new__(cls, name, bases, kwargs)
        if title is not None:
            obj._metadata_ = DatasetMetadata.from_title(title)

        return obj

    @property
    def metadata(cls) -> DatasetMetadata:
        """
        Returns the metadata associated with the Enum class.

        Returns:
            An instance of DatasetMetadata that contains metadata for the Enum class.
        """
        return cls._metadata_

    def to_dataframe(cls) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the metadata for the Enum class.

        Returns:
            A pandas DataFrame containing the metadata for the Enum class.
        """
        df = cls.md.to_dataframe()
        df.index = pd.Index(df["standard_name"].map(lambda x: cls(x).name).rename("member_name"))

        return df


class CFDatasetEnum(StrEnum, metaclass=CFDatasetEnumMeta):
    @property
    def dvars(self) -> types.MappingProxyType[str, Any]:
        return self._metadata_.variables[self]

    @property
    def crs(self) -> pyproj.CRS:
        return self._metadata_.crs

    @property
    def short_name(self) -> str:
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
