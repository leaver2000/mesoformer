from __future__ import annotations

import abc

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.coordinates import DatasetCoordinates

from ..generic import Data
from ..typing import Hashable, Iterable, ListLike, NDArray, Number
from ..utils import log_scale, sort_unique
from .metadata import (
    COORDINATES,
    DIMENSIONS,
    LAT,
    LON,
    LVL,
    TIME,
    Coordinates,
    Dimensions,
    T,
    X,
    Y,
    Z,
)

P0 = 1013.25  # - mbar
P1 = 25.0  # - mbar
ERA5_GRID_RESOLUTION = 30.0  # km / px
RATE = ERA5_GRID_RESOLUTION / 2
URMA_GRID_RESOLUTION = 2.5  # km / px
MESOSCALE_BETA = 200.0  # km
DEFAULT_PRESSURE: ListLike[Number] = [925.0, 850.0, 700.0, 500.0, 300.0]


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
        self._tropo = tropo = self._sort_unique_descending(troposphere if troposphere is not None else self._arange())
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

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()).set_index("hpa").sort_index()


# =====================================================================================================================
def check_independent_dims(dims: Iterable[Hashable]) -> bool:
    return tuple(dims) == DIMENSIONS


def check_independent_coords(coords: DatasetCoordinates) -> bool:
    return set(coords) == set(COORDINATES) and all(coords[x].dims == (Y, X) for x in (LON, LAT))


def is_independent(ds: xr.Dataset) -> bool:
    return check_independent_dims(ds.dims) and check_independent_coords(ds.coords)


def set_independent(ds: xr.Dataset) -> xr.Dataset:
    """insures a dependant dataset is in the correct format."""
    if is_independent(ds):
        return ds

    # - move any coordinates that may be assigned as variables to coordinates
    ds = ds.set_coords(Coordinates.intersection(ds.variables))

    # - rename the dims and variables
    ds = ds.rename_dims(Dimensions.map(ds.dims))
    ds = ds.rename_vars(Coordinates.map(ds.coords))

    # - Move any coordinates that may be assigned as variables to coordinates
    lon, lat = (ds[coord].compute() for coord in (LON, LAT))
    ds[LON] = lon
    ds[LAT] = lat

    # - dimension assignment
    if missing := Dimensions.difference(ds.dims):
        for dim in missing:
            ds = ds.expand_dims(dim, axis=[DIMENSIONS.index(dim)])

    # - coordinate assignment
    if missing := Coordinates.difference(ds.coords):
        ds = ds.assign_coords({coord: (coord.axis, ["derived"]) for coord in missing})

    if ds[LAT].dims == (Y,) and ds[LON].dims == (X,):
        # 5.2. Two-Dimensional Latitude, Longitude, Coordinate
        # Variables
        # The latitude and longitude coordinates of a horizontal grid that was not defined as a Cartesian
        # product of latitude and longitude axes, can sometimes be represented using two-dimensional
        # coordinate variables. These variables are identified as coordinates by use of the coordinates
        # attribute
        lon, lat = (ds[coord].to_numpy() for coord in (LON, LAT))
        yy, xx = np.meshgrid(lat, lon, indexing="xy")

        ds = ds.assign_coords({LAT: (LAT.axis, yy), LON: (LON.axis, xx)})

    ds = ds.transpose(*DIMENSIONS)
    assert is_independent(ds)
    return ds


class IndependentABC(abc.ABC):
    @property
    @abc.abstractmethod
    def ds(self) -> xr.Dataset:
        ...

    def to_array(self):
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

    @property
    def lons(self) -> xr.DataArray:
        return self.ds[LON]

    @property
    def lats(self) -> xr.DataArray:
        return self.ds[LAT]

    def __repr__(self) -> str:
        return self.ds.__repr__()

    def _repr_html_(self) -> str:
        return self.ds._repr_html_()
