import random
from typing import TypeVar

import mesoscaler as ms
import numpy as np
import torch
import torch.utils.data
from mesoscaler._typing import Array, Nt, Nv, Nx, Ny, Nz
from mesoscaler.enums import (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE,
    TEMPERATURE_2M,
    U_COMPONENT_OF_WIND,
    U_WIND_COMPONENT_10M,
    V_COMPONENT_OF_WIND,
    V_WIND_COMPONENT_10M,
)

era5_dvars = [
    GEOPOTENTIAL,
    TEMPERATURE,
    SPECIFIC_HUMIDITY,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
]

urma_dvars = [
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
]
assert len(urma_dvars) == len(era5_dvars)
CHANNELS = len(urma_dvars)

_T = TypeVar("_T")


class IterableDataset(torch.utils.data.IterableDataset[_T]):
    def __init__(self, channel_size: int) -> None:
        super().__init__()
        self.channel_size = channel_size


class MesoscalerDataset(torch.utils.data.IterableDataset[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]):
    def __init__(self, resampler: ms.ReSampler, time_batch_size: int) -> None:
        super().__init__()

        self.resampler = resampler
        self.indices = ms.AreaOfInterestSampler(
            resampler.domain, aoi=(-106.6, 25.8, -93.5, 36.5), time_batch_size=time_batch_size
        )
        torch.random.manual_seed(0)
        random.shuffle(self.indices.indices)

    def __iter__(self):
        for (lon, lat), time in self.indices:
            yield self.resampler(lon, lat, time)

    def __len__(self):
        return len(self.indices)
