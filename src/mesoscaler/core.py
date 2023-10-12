from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pyproj
import pyresample.geometry
import xarray as xr
from xarray.core.coordinates import DatasetCoordinates

from .enums import (
    DIMENSIONS,
    LAT,
    LON,
    LVL,
    TIME,
    Coordinates,
    DependentVariables,
    Dimensions,
    T,
    X,
    Y,
    Z,
)
from .generic import Data
from .typing import (
    N2,
    N4,
    Any,
    Array,
    Callable,
    Hashable,
    Iterable,
    Iterator,  # type:ignore
    ListLike,
    Literal,  # type:ignore
    Mapping,
    N,
    NDArray,
    Number,
    Self,
    Slice,
    TypeAlias,
    Sequence,  # noqa
)
from .utils import log_scale, sort_unique

AreaExtent: TypeAlias = Array[[N4], np.float_]  # type:ignore
Definition: TypeAlias = pyresample.geometry.BaseDefinition
Depends: TypeAlias = "type[DependentVariables] | DependentVariables | Sequence[DependentVariables] | Dependencies"
ResampleInstruction: TypeAlias = "tuple[DependentDataset, AreaExtent]"
Unit = Literal["km", "m"]
# =====================================================================================================================
P0 = 1013.25  # - mbar
P1 = 25.0  # - mbar
ERA5_GRID_RESOLUTION = 30.0  # km / px
# RATE = ERA5_GRID_RESOLUTION / 2
URMA_GRID_RESOLUTION = 2.5  # km / px
MESOSCALE_BETA = 200.0  # km
DEFAULT_PRESSURE: ListLike[Number] = [P0, 925.0, 850.0, 700.0, 500.0, 300.0]
DERIVED_SFC_COORDINATE = {LVL: (LVL.axis, [P0])}
_units: Mapping[Unit, float] = {"km": 1.0, "m": 1000.0}
CHANNELS = "channels"
VARIABLES = "variable"
_GRID_DEFINITION = "grid_definition"
_DEPENDS = "depends"


# =====================================================================================================================
# - tools for preparing to data into a common format and convention
# =====================================================================================================================
class Dependencies:
    @staticmethod
    def _validate_variables(depends: Depends) -> tuple[type[DependentVariables], list[DependentVariables]]:
        if isinstance(depends, Dependencies):
            return depends.enum, depends.depends
        elif isinstance(depends, type):
            assert issubclass(depends, DependentVariables)
            enum = depends
            depends = list(depends)  # type: ignore
        elif isinstance(depends, DependentVariables):
            enum = depends.__class__
            depends = [depends]
        else:
            enum = depends[0].__class__
            depends = list(depends)

        for dvar in depends:
            assert isinstance(dvar, enum)
        return enum, depends

    def __init__(self, depends: Depends):
        self.enum, self.depends = self._validate_variables(depends)

    @property
    def difference(self) -> set[DependentVariables]:
        return self.enum.difference(self.depends)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.enum.__name__})"

    @property
    def names(self) -> pd.Index[str]:
        return self.enum._names

    @property
    def crs(self) -> pyproj.CRS:
        return self.enum.crs  # type: ignore

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.enum.metadata  # type: ignore


def is_dimension_independent(dims: Iterable[Hashable]) -> bool:
    return all(isinstance(dim, Dimensions) for dim in dims) and set(dims) == set(Dimensions)


def is_coordinate_independent(coords: DatasetCoordinates) -> bool:
    return (
        all(isinstance(coord, Coordinates) for coord in coords)
        and set(coords) == set(Coordinates)
        and all(coords[x].dims == (Y, X) for x in (LON, LAT))
    )


def is_independent(ds: xr.Dataset) -> bool:
    return is_coordinate_independent(ds.coords) and is_dimension_independent(ds.dims)


def make_independent(ds: xr.Dataset) -> xr.Dataset:  # type:ignore
    """insures a dependant dataset is in the correct format."""
    if is_independent(ds):
        return ds
    #  - rename the dims and coordinates
    ds = ds.rename_dims(Dimensions.remap(ds.dims)).rename_vars(Coordinates.remap(ds.coords))
    # - move any coordinates assigned as variables to the coordinates
    ds = ds.set_coords(Coordinates.intersection(ds.variables))
    ds = ds.rename_vars(Coordinates.remap(ds.coords))

    ds[LON], ds[LAT] = (ds[coord].compute() for coord in (LON, LAT))

    # - dimension assignment
    if missing_dims := Dimensions.difference(ds.dims):
        for dim in missing_dims:
            ds = ds.expand_dims(dim, axis=[DIMENSIONS.index(dim)])

    # # - coordinate assignment
    if missing_coords := Coordinates.difference(ds.coords):
        if missing_coords != {LVL}:
            raise ValueError(f"missing coordinates {missing_coords}; only {LVL} is allowed to be missing!")
        ds = ds.assign_coords(DERIVED_SFC_COORDINATE).set_xindex(LVL)

    if ds[LAT].dims == (Y,) and ds[LON].dims == (X,):
        # 5.2. Two-Dimensional Latitude, Longitude, Coordinate
        # Variables
        # The latitude and longitude coordinates of a horizontal grid that was not defined as a Cartesian
        # product of latitude and longitude axes, can sometimes be represented using two-dimensional
        # coordinate variables. These variables are identified as coordinates by use of the coordinates
        # attribute
        lon, lat = (ds[coord].to_numpy() for coord in (LON, LAT))
        yy, xx = np.meshgrid(lat, lon, indexing="ij")

        ds = ds.assign_coords({LAT: (LAT.axis, yy), LON: (LON.axis, xx)})

    ds = ds.transpose(*DIMENSIONS)

    return ds


# =====================================================================================================================
#
# =====================================================================================================================
class IndependentDataset(xr.Dataset):
    __slots__ = ()

    @classmethod
    def from_dependant(cls, ds: xr.Dataset, **kwargs) -> Self:
        return cls(make_independent(ds), **kwargs)

    # - dims
    @property
    def t(self) -> xr.DataArray:
        return self[T]

    @property
    def z(self) -> xr.DataArray:
        return self[Z]

    @property
    def y(self) -> xr.DataArray:
        return self[Y]

    @property
    def x(self) -> xr.DataArray:
        return self[X]

    # - coords
    @property
    def time(self) -> xr.DataArray:
        return self[TIME]

    @property
    def level(self) -> xr.DataArray:
        return self[LVL]

    @property
    def lats(self) -> xr.DataArray:
        return self[LAT]

    @property
    def lons(self) -> xr.DataArray:
        return self[LON]


class DependentDataset(IndependentDataset):
    __slots__ = ()
    __dims__ = (T, Z, Y, X)
    __coords__ = (TIME, LVL, LAT, LON)

    def __init__(
        self,
        data: xr.Dataset,
        *,
        depends: Depends | None = None,
        attrs: Mapping[str, Any] | None = None,
    ):
        if isinstance(data, DependentDataset) and attrs is None:
            attrs = {
                _GRID_DEFINITION: data.grid_definition,
                _DEPENDS: data.depends,
            }
        super().__init__(data, attrs=attrs)
        if depends is not None:
            self.attrs[_DEPENDS] = depends

        if _GRID_DEFINITION not in self.attrs:
            lons, lats = (self[x].to_numpy() for x in (LON, LAT))
            lons = (lons + 180.0) % 360 - 180.0
            self.attrs[_GRID_DEFINITION] = pyresample.geometry.GridDefinition(lons, lats)
        assert is_independent(self)

    @classmethod
    def from_zarr(
        cls,
        store: Any,
        depends: Depends,
    ) -> DependentDataset:
        depends = Dependencies(depends)
        return cls.from_dependant(xr.open_zarr(store, drop_variables=depends.difference), depends=depends)

    @property
    def grid_definition(self) -> pyresample.geometry.GridDefinition:
        return self.attrs[_GRID_DEFINITION]

    @property
    def depends(self) -> Dependencies:
        return self.attrs[_DEPENDS]


# =====================================================================================================================
# Resampling
# =====================================================================================================================


def _instruction_iterator(scale: Mesoscale, *dsets: DependentDataset) -> Iterator[ResampleInstruction]:
    levels = np.concatenate([ds.level for ds in dsets])
    levels = np.sort(levels[np.isin(levels, scale.hpa)])[::-1]
    datasets = (ds.sel({LVL: [lvl]}) for lvl in levels for ds in dsets if lvl in ds.level)
    extents = scale.stack_extent() * 1000.0  # km -> m
    return zip(datasets, extents)


class Mesoscale(Data[NDArray[np.float_]]):
    def __init__(
        self,
        dx: float = 200.0,
        dy: float | None = None,
        *,
        rate: float = 1.0,
        pressure: ListLike[Number] = DEFAULT_PRESSURE,
        troposphere: ListLike[Number] | None = None,
    ) -> None:
        super().__init__()
        tropo = self._sort_unique_descending(troposphere if troposphere is not None else self._arange())
        self._hpa = hpa = self._sort_unique_descending(pressure)
        if not all(np.isin(hpa, tropo)):
            raise ValueError(f"pressure {hpa} must be a subset of troposphere {tropo}")
        mask = np.isin(tropo, hpa)
        self._scale = scale = log_scale(tropo, rate=rate)[::-1][mask]  # ascending scale
        self._dx, self._dy = scale[np.newaxis] * np.array([[dx], [dy or dx]])  # ascending extent km scale

    @staticmethod
    def _sort_unique_descending(x: ListLike[Number]) -> NDArray[np.float_]:
        return sort_unique(x)[::-1].astype(np.float_)

    @staticmethod
    def _arange(
        start: int = 1000,
        stop: int = 25 - 1,
        step: int = -25,
        *,
        p0: float = P0,
        p1=P1,
    ) -> ListLike[Number]:
        return [p0, *range(start, stop, step), p1]

    @classmethod
    def arange(
        cls,
        dx: float = 200.0,
        dy: float | None = None,
        start: int = 1000,
        stop: int = 25 - 1,
        step: int = -25,
        *,
        p0: float = P0,
        p1=P1,
        rate: float = 1.0,
        pressure: ListLike[Number],
    ) -> Mesoscale:
        return cls(dx, dy, rate=rate, pressure=pressure, troposphere=cls._arange(start, stop, step, p0=p0, p1=p1))

    @property
    def hpa(self) -> NDArray[np.float_]:
        return self._hpa

    @property
    def scale(self) -> NDArray[np.float_]:
        return self._scale

    @property
    def dx(self) -> NDArray[np.float_]:
        return self._dx

    @property
    def dy(self) -> NDArray[np.float_]:
        return self._dy

    @property
    def data(self) -> Iterable[tuple[str, NDArray[np.float_]]]:
        yield from (("scale", self.scale), ("hpa", self.hpa), ("dx", self.dx), ("dy", self.dy))

    def __array__(self) -> Array[[N, N2], np.float_]:
        return self.to_numpy()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()).set_index("hpa").sort_index()

    def to_numpy(self, *, units: Unit = "km") -> Array[[N, N2], np.float_]:
        return np.c_[self.dx, self.dy] * _units[units]

    def stack_extent(self, *, units: Unit = "km") -> Array[[N, N4], np.float_]:
        xy = self.to_numpy(units=units)
        return np.c_[-xy, xy]

    def resample(self, *dsets: DependentDataset, height: int = 80, width: int = 80) -> ReSampleGenerator:
        return ReSampleGenerator(_instruction_iterator(self, *dsets), height=height, width=width)


_laea = {
    "proj": "laea",
    "x_0": 0,
    "y_0": 0,
    "ellps": "WGS84",
    "units": "m",
    "no_defs": None,
    "type": "crs",
}


def lambert_equal_area(longitude: float, latitude: float) -> pyproj.CRS:
    return pyproj.CRS(_laea | {"lat_0": latitude, "lon_0": longitude})


class ReSampleGenerator:
    def __init__(
        self,
        it: Iterator[ResampleInstruction],
        /,
        *,
        height: int = 80,
        width: int = 80,
    ) -> None:
        self._instruction = iter(it)
        self.height = height
        self.width = width

    @classmethod
    def create(
        cls, scale: Mesoscale, *dsets: DependentDataset, height: int = 80, width: int = 80
    ) -> ReSampleGenerator:
        return cls(_instruction_iterator(scale, *dsets), height=height, width=width)

    def _generate(
        self,
        __func: Callable[[Definition, xr.Dataset, Definition], NDArray],
        partial: functools.partial[pyresample.geometry.AreaDefinition],
        time: Slice[np.datetime64],
    ) -> Array[[N, N, N, N, N], np.float_]:
        # - resample the data
        arr = np.stack(
            [
                __func(ds.grid_definition, ds.sel({TIME: time}), partial(area_extent=area_extent))
                for ds, area_extent in self._instruction
            ],
        )
        # - reshape the data
        t = (time.stop - time.start) // np.timedelta64(1, "h") + 1
        z, y, x = arr.shape[:3]
        arr = arr.reshape((z, y, x, t, -1))  # unsqueeze C
        return np.moveaxis(arr, (-1, -2), (0, 1))

    def nearest(
        self,
        longitude: float,
        latitude: float,
        *,
        time: Slice[np.datetime64],
    ) -> Array[[N, N, N, N, N], np.float_]:
        partial = functools.partial(
            pyresample.geometry.AreaDefinition,
            "",
            "",
            "",
            width=self.width,
            height=self.height,
            projection=lambert_equal_area(longitude, latitude),
        )
        return self._generate(self._resample_nearest, partial, time)

    def _resample_nearest(
        self,
        source: pyresample.geometry.BaseDefinition,
        ds: xr.Dataset,
        target: pyresample.geometry.BaseDefinition,
    ) -> NDArray[np.float_]:
        return pyresample.kd_tree.resample_nearest(
            source,
            data=ds.transpose(Y, X, ...).to_array("V").stack({"C": ["V", T, Z]}).to_numpy(),
            target_geo_def=target,
            radius_of_influence=500000,
        )
