"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

__all__ = ["auto_field", "TableEnum", "IndependentVariables", "DependentVariables"]
import collections
import enum
import types
from typing import MutableMapping, NamedTuple, TypeAlias

import pandas as pd
import pyproj

from .generic import Loc as _Loc
from .typing import (
    Any,
    Hashable,
    HashableT,
    Iterable,
    Mapping,
    MutableMapping,
    NamedTuple,
    TypeAlias,
    TypeVar,
)
from .utils import is_scalar

LOC = "__mesometa_loc__"
CLASS_METADATA = "__mesometa_cls_data__"
MEMBER_METADATA = "__mesometa_member_data__"
MEMBER_ALIASES = "__mesometa_member_aliases__"
MEMBER_SERIES = "__mesometa_series__"

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


_T = TypeVar("_T")


# =====================================================================================================================
class _Field(NamedTuple):
    value: Any
    metadata: Mapping[str, Any]


def auto_field(value: _T | Any = None, *, aliases: list[_T] | None = None, **metadata: Any) -> Any:
    if value is None:
        value = enum.auto()
    if MEMBER_ALIASES in metadata and aliases is None:
        assert isinstance(metadata[MEMBER_ALIASES], list)
    elif MEMBER_ALIASES in metadata and aliases is not None:
        raise ValueError("Field metadata contains aliases and aliases were passed as an argument.")
    else:
        metadata[MEMBER_ALIASES] = aliases or []

    return _Field(value, metadata)


def _generate_metadata() -> Iterable[dict[str, Any]]:
    metadata = getattr(_EnumMetaCls.__metadata__, "_data")  # type: Mapping[str, types.MappingProxyType[str, Any]]
    for key, value in metadata.items():
        data = {"hash": key} | dict(value)  # type: dict[str, Any]
        data[MEMBER_SERIES] = dict(data.pop(MEMBER_SERIES))
        data[MEMBER_METADATA] = dict(data.pop(MEMBER_METADATA))
        data[MEMBER_ALIASES] = data.pop(MEMBER_ALIASES).to_dict(orient="records")
        yield data


def get_metadata() -> list[dict[str, Any]]:
    return list(_generate_metadata())


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
    aliases = pd.DataFrame.from_dict(
        {k: list(set(metadata[k].pop(MEMBER_ALIASES, []))) for k in member_map.keys()}, orient="index"
    ).T

    member_metadata = types.MappingProxyType(collections.defaultdict(dict, metadata))
    member_series = pd.Series(
        list(member_map.values()),
        name=name,
        index=pd.Index(list(member_map.keys()), name="member_names", dtype="string"),
    )
    return types.MappingProxyType(
        {
            "name": name,
            CLASS_METADATA: class_metadata,
            MEMBER_METADATA: member_metadata,
            MEMBER_ALIASES: aliases,
            MEMBER_SERIES: member_series,
            LOC: _Loc(list, member_series),
        }
    )


# =====================================================================================================================
class Descriptor:
    __getitem__: Any
    _data: collections.defaultdict[int, Mapping[str, Any]]

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


class _EnumMetaCls(enum.EnumMeta):
    __metadata__ = Descriptor()

    def __new__(cls, name: str, bases: tuple[Any, ...], cls_dict: enum._EnumDict, **kwargs: Any):
        cls_dict, member_metadata = _unpack_info(cls_dict)
        obj = super().__new__(cls, name, bases, cls_dict)
        if obj._member_names_:
            obj.__metadata__ = _repack_info(name, obj._member_map_, member_metadata, kwargs)

        return obj

    # =================================================================================================================
    @property
    def metadata(cls) -> MutableMapping[str, Any]:
        return cls.__metadata__[CLASS_METADATA]

    @property
    def loc(cls) -> _Loc[str | bool | slice, list["TableEnum"]]:
        return cls.__metadata__[LOC]

    @property
    def _series(cls) -> pd.Series:
        return cls.__metadata__[MEMBER_SERIES]

    @property
    def _names(cls) -> pd.Index[str]:
        return cls._series.index

    @property
    def _member_metadata(cls) -> types.MappingProxyType[str, MemberMetadata]:
        return cls.__metadata__[MEMBER_METADATA]

    @property
    def _aliases(cls) -> pd.DataFrame:
        return cls.__metadata__[MEMBER_ALIASES]

    # =================================================================================================================
    # - metadata properties

    def to_frame(cls) -> pd.DataFrame:
        df = cls._aliases.copy()

        return df

    def to_series(cls) -> pd.Series[Any]:
        return pd.Series(cls._member_map_, name=cls.__name__)

    # =================================================================================================================

    def __call__(cls, __items: Iterable[Hashable] | Hashable) -> Any | list[Any]:  # type: ignore[override]
        """It is possible to return multiple members if the members share an alias."""
        if is_scalar(__items):
            return cls._get_from_hashable(__items)  # type: ignore

        return cls.intersection(__items)

    def _get_from_hashable(cls, __item: Hashable, /) -> Any:
        s = cls._series
        if isinstance(__item, str) and (member := s.get(__item, None)):
            return member
        elif (mask := s == __item).any():
            return s[mask].item()

        return s[cls.is_in(__item)].item()

    def to_list(cls, /) -> list[Any]:
        return cls._series.to_list()

    def is_in(cls, __x: Hashable | Iterable[Hashable], /) -> pd.Series[bool]:
        if isinstance(__x, Hashable):
            __x = [__x]
        return cls._aliases.isin(__x).any(axis=0, skipna=True) | cls._names.isin(__x)

    def difference(cls, __x: Hashable | Iterable[Hashable], /) -> list[Any]:
        mask = ~cls.is_in(__x)
        return cls.loc[mask]  # type: ignore

    def intersection(cls, __x: Hashable | Iterable[Hashable], /):
        mask = cls.is_in(__x)
        return cls._series[mask].to_list()

    def remap(cls, __x: Iterable[HashableT], /):
        return {x: cls.__call__(x) for x in __x}


class TableEnum(enum.Enum, metaclass=_EnumMetaCls):
    @property
    def aliases(self) -> list[Any]:
        return self.__class__._aliases[self.name].dropna().to_list()

    @property
    def metadata(self) -> MemberMetadata:
        return self.__class__._member_metadata[self.name]


class IndependentVariables(str, TableEnum):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        return name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


def get_crs(name: str) -> pyproj.CRS:
    # TODO: move this to disk
    if name == "ERA5":
        cf = {
            "crs_wkt": 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
            "geographic_crs_name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "World Geodetic System 1984 ensemble",
            "grid_mapping_name": "latitude_longitude",
        }
    elif name == "URMA":
        cf = {
            "geographic_crs_name": "NDFD CONUS 2.5km Lambert Conformal Conic",
            "projected_crs_name": "NDFD",
            "grid_mapping_name": "lambert_conformal_conic",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31424518,
            "inverse_flattening": 298.25722356301,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "WGS84",
            "latitude_of_projection_origin": 20.191999,
            "longitude_of_projection_origin": 238.445999,
            "standard_parallel": 25,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "units": "m",
        }
    else:
        raise ValueError(f"Unknown CRS {name!r}")
    return pyproj.CRS.from_cf(cf)


class DependentVariables(IndependentVariables):
    @classmethod  # type:ignore
    @property
    def crs(cls) -> pyproj.CRS:
        md = cls.metadata  # type: MutableMapping[str, Any] # type: ignore
        if not (crs := md.get("crs", None)):
            crs = md["crs"] = get_crs(cls.__name__)

        return crs
