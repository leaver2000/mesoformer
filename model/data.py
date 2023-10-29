from torch.utils.data import IterableDataset
import mesoscaler as ms
from mesoscaler._typing import Array, Nv, Nt, Nz, Ny, Nx
import numpy as np

from mesoscaler.enums import (
    # - ERA5
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    # - URMA
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
    SURFACE_PRESSURE,
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

#


class Dataset(IterableDataset[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]):
    def __init__(self, resampler: ms.ReSampler) -> None:
        super().__init__()

        self.resampler = resampler
        self.indices = ms.AreaOfInterestSampler(
            resampler.domain,
            aoi=(-106.6, 25.8, -93.5, 36.5),
        )

    def __iter__(self):
        for (lon, lat), time in self.indices:
            yield self.resampler(lon, lat, time)


def load_data(urma_store: str, era5_store: str) -> ms.DatasetSequence:
    return ms.open_datasets([(urma_store, urma_dvars), (era5_store, era5_dvars)])


def data_loader(
    dataset_sequence: ms.DatasetSequence,
    width=80,
    height=40,
    distance_ratio=2.5,  # km
    patch_ratio=0.2,
    levels=[1013.25, 1000, 925, 850],
    aoi=(-106.6, 25.8, -93.5, 36.5),
):
    patch_size = (int(width * patch_ratio), int(height * patch_ratio))
    dx = int(width * distance_ratio)
    dy = int(height * distance_ratio)

    scale = ms.Mesoscale(dx, dy, levels=levels)

    ds = Dataset(scale.resample(dataset_sequence, height=height, width=width))
