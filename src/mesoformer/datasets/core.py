from __future__ import annotations

import numpy as np
import pandas as pd

from ..generic import Contents
from ..typing import Iterable, NDArray, Number, SequenceLike
from ..utils import log_scale, sort_unique

P0 = 1013.25  # - mbar
P1 = 25.0  # - mbar
ERA5_GRID_RESOLUTION = 30.0  # km / px
RATE = ERA5_GRID_RESOLUTION / 2
URMA_GRID_RESOLUTION = 2.5  # km / px
MESOSCALE_BETA = 200.0  # km


def arange_troposphere(
    start: int = 1000,
    stop: int = 25 - 1,
    step: int = -25,
    *,
    p0: float = P0,
    p1=P1,
) -> NDArray[np.float_]:
    x = sort_unique([p0, *range(start, stop, step), p1])[::-1] # descending pressure
    return x


class Mesoscale(Contents[NDArray[np.float_]]):
    def __init__(
        self,
        dx: float = 200.0,
        dy: float | None = None,
        *,
        rate: float = 1.0,
        pressure: SequenceLike[Number],
        troposphere: NDArray[np.float_] | None = None,
    ) -> None:
        super().__init__()
        self._tropo = tropo = sort_unique(troposphere)[::-1] if troposphere is not None else arange_troposphere() # descending pressure
        self._hpa = hpa = sort_unique(pressure)[::-1]

        mask = np.isin(tropo, hpa)
        self._scale = scale = log_scale(tropo, rate=rate)[::-1][mask]
        self._dx, self._dy = scale[np.newaxis] * np.array([[dx], [dy or dx]])

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

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()).set_index("hPa").sort_index()

    @property
    def content(self) -> Iterable[tuple[str, NDArray[np.float_]]]:
        return [
            ("scale", self.scale),
            ("hPa", self.hpa),
            ("dx", self.dx),
            ("dy", self.dy),
            ("era5_px", self.dx / ERA5_GRID_RESOLUTION),
            ("era5_py", self.dy / ERA5_GRID_RESOLUTION),
            ("urma_px", self.dx / URMA_GRID_RESOLUTION),
            ("urma_py", self.dy / URMA_GRID_RESOLUTION),
        ]
