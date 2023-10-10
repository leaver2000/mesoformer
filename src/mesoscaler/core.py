from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj
import pyresample.geometry
import xarray as xr
from xarray.core.coordinates import DatasetCoordinates

from .enums import (
    COORDINATES,
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
    Final,
    Hashable,
    Iterable,
    ListLike,
    Literal,
    Mapping,
    N,
    NDArray,
    Number,
    Sequence,
    StrPath,
)
from .utils import log_scale, sort_unique

P0 = 1013.25  # - mbar
P1 = 25.0  # - mbar
ERA5_GRID_RESOLUTION = 30.0  # km / px
# RATE = ERA5_GRID_RESOLUTION / 2
URMA_GRID_RESOLUTION = 2.5  # km / px
MESOSCALE_BETA = 200.0  # km
DEFAULT_PRESSURE: ListLike[Number] = [925.0, 850.0, 700.0, 500.0, 300.0]

DERIVED_SFC_COORDINATE = {LVL: (LVL.axis, [P0])}

Unit = Literal["km", "m"]
_units: Mapping[Unit, float] = {"km": 1.0, "m": 1000.0}


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

    def to_area_extents(self, *, units: Unit = "km") -> Array[[N, N4], np.float_]:
        xy = self.to_numpy(units=units)
        return np.c_[-xy, xy]


# =====================================================================================================================
def is_dimension_independent(dims: Iterable[Hashable]) -> bool:
    return all(isinstance(dim, Dimensions) for dim in dims) and tuple(dims) == DIMENSIONS


def is_coordinate_independent(coords: DatasetCoordinates) -> bool:
    return (
        all(isinstance(coord, Coordinates) for coord in coords)
        and set(coords) == set(COORDINATES)
        and all(coords[x].dims == (Y, X) for x in (LON, LAT))
    )


def is_independent(ds: xr.Dataset) -> bool:
    return is_dimension_independent(ds.dims) and is_coordinate_independent(ds.coords)


def make_independent(ds: xr.Dataset) -> xr.Dataset:  # type:ignore
    """insures a dependant dataset is in the correct format."""
    if is_independent(ds):
        return ds
    Dimensions.remap(ds.dims)

    ds = ds.rename_dims(Dimensions.remap(ds.dims))
    ds = ds.rename_vars(Coordinates.remap(ds.coords))
    ds = ds.set_coords(Coordinates.intersection(ds.variables))
    ds = ds.rename_vars(Coordinates.remap(ds.coords))
    ds = ds.rename_vars(Coordinates.remap(ds.coords))

    ds[LON], ds[LAT] = (ds[coord].compute() for coord in (LON, LAT))

    # - dimension assignment
    if missing_dims := Dimensions.difference(ds.dims):
        for dim in missing_dims:
            ds = ds.expand_dims(dim, axis=[DIMENSIONS.index(dim)])

    # # - coordinate assignment
    if missing_coords := Coordinates.difference(ds.coords):
        assert missing_coords == [LVL], missing_coords
        ds = ds.assign_coords(DERIVED_SFC_COORDINATE)

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
    assert is_independent(ds)
    return ds


Depends = type[DependentVariables] | DependentVariables | Sequence[DependentVariables]


# =====================================================================================================================
class GriddedDataset(pyresample.geometry.GridDefinition):
    def __init__(self, ds: xr.Dataset, dvars: Depends) -> None:
        if not is_independent(ds):
            raise ValueError("Dataset must be independent")
        lons, lats = (ds[x].to_numpy() for x in (LON, LAT))
        lons = (lons + 180.0) % 360 - 180.0
        super().__init__(lons, lats)
        enum, dvars = self._validate_variables(dvars)
        self.enum: Final = enum
        self.dvars: Final = dvars
        self.ds: Final = ds[dvars]

    @staticmethod
    def _validate_variables(dvars: Depends) -> tuple[type[DependentVariables], list[DependentVariables]]:
        if isinstance(dvars, type):
            assert issubclass(dvars, DependentVariables)
            enum = dvars
            dvars = list(dvars)  # type: ignore
        elif isinstance(dvars, DependentVariables):
            enum = dvars.__class__
            dvars = [dvars]
        else:
            enum = dvars[0].__class__
            dvars = list(dvars)

        for dvar in dvars:
            assert isinstance(dvar, enum)
        return enum, dvars

    @property
    def names(self) -> pd.Index[str]:
        return self.enum._names

    @property
    def crs(self) -> pyproj.CRS:
        return self.enum.crs

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.enum.metadata

    @classmethod
    def from_zarr(cls, path: StrPath, dvars: Depends) -> GriddedDataset:
        return cls.from_dependant(xr.open_zarr(path), dvars)

    @classmethod
    def from_dependant(cls, ds: xr.Dataset, dvars: Depends) -> GriddedDataset:
        return cls(make_independent(ds), dvars)

    def to_array(self) -> xr.DataArray:
        return self.ds.to_array().transpose(X, Y, ...)

    # - dims
    @property
    def t(self) -> xr.DataArray:
        return self.ds[T]

    @property
    def z(self) -> xr.DataArray:
        return self.ds[Z]

    @property
    def y(self) -> xr.DataArray:
        return self.ds[Y]

    @property
    def x(self) -> xr.DataArray:
        return self.ds[X]

    # - coords
    @property
    def time(self) -> xr.DataArray:
        return self.ds[TIME]

    @property
    def level(self) -> xr.DataArray:
        return self.ds[LVL]

    def __repr__(self) -> str:
        return self.ds.__repr__()

    def _repr_html_(self) -> str:
        return self.ds._repr_html_()

    # def get_area_definition(self) -> pyresample.geometry.AreaDefinition:
    #     raise NotImplementedError()
    #     crs = self.metadata.crs
    #     area_def = pyresample.geometry.AreaDefinition(
    #         "area_def",
    #         "area_def",
    #         "area_def",
    #         projection=crs,
    #         width=self.x.size,
    #         height=self.y.size,
    #     )
    #     return area_def

    # def ge_T_coordinates(self: IndependentDataset) -> tuple[Array[[N, N], np.float_], Array[[N, N], np.float_]]:
    #     lons = self.lons.to_numpy()
    #     lats = self.lats.to_numpy()
    #     lons = (lons + 180.0) % 360 - 180.0
    #     return lons, lats

    # def get_grid_definition(self: IndependentDataset) -> pyresample.geometry.GridDefinition:
    #     lons, lats = self.ge_T_coordinates()

    #     return pyresample.geometry.GridDefinition(lons, lats)
