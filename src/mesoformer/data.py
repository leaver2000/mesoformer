from typing import TypedDict

import numpy as np
from mesoscaler._typing import Array, N, StrPath, ListLike, Number
from mesoscaler.core import ArrayWorker, DependentDataset, Depends, Mesoscale, DEFAULT_PRESSURE
from mesoscaler.enums import (  # GEOPOTENTIAL,; SPECIFIC_HUMIDITY,; TEMPERATURE,; U_COMPONENT_OF_WIND,; V_COMPONENT_OF_WIND,; VERTICAL_VELOCITY,
    CoordinateReferenceSystem,
    LiteralCRS,
)
from mesoscaler.generic import DataConsumer
from torch.utils.data import DataLoader


class Config(TypedDict):
    store: StrPath
    depends: Depends


class Sampler:
    ...


def worker(
    dx: float = 200,  # km
    dy: float = 200,  # km
    *dsets: Config | DependentDataset,
    height: int = 80,  # px
    width: int = 80,  # px
    rate: int = 15,
    target_projection: LiteralCRS = CoordinateReferenceSystem.lambert_azimuthal_equal_area,
    # rate: float = 1,
    pressure: ListLike[Number] = DEFAULT_PRESSURE,
    troposphere: ListLike[Number] | None = None,
) -> ArrayWorker:
    return ArrayWorker(
        [],
        *(ds if isinstance(ds, DependentDataset) else DependentDataset.from_zarr(**ds) for ds in dsets),
        scale=Mesoscale(dx, dy, rate=rate, pressure=pressure, troposphere=troposphere),
        height=height,
        width=width,
        target_projection=target_projection,
    )


def consumer(
    dx: float = 200,  # km
    dy: float = 200,  # km
    *dsets: Config | DependentDataset,
    height: int = 80,  # px
    width: int = 80,  # px
    rate: int = 15,
    target_projection: LiteralCRS = CoordinateReferenceSystem.lambert_azimuthal_equal_area,
    pressure: ListLike[Number] = DEFAULT_PRESSURE,
    troposphere: ListLike[Number] | None = None,
    maxsize: int = 1,
    timeout: float | None = None,
):
    return DataConsumer(
        worker(
            dx,
            dy,
            *dsets,
            height=height,
            width=width,
            rate=rate,
            target_projection=target_projection,
            pressure=pressure,
            troposphere=troposphere,
        ),
        maxsize=maxsize,
        timeout=timeout,
    )


def loader(
    # - dataset
    dx: float = 200,  # km
    dy: float = 200,  # km
    *dsets: Config | DependentDataset,
    height: int = 80,  # px
    width: int = 80,  # px
    rate: int = 15,
    target_projection: LiteralCRS = CoordinateReferenceSystem.lambert_azimuthal_equal_area,
    pressure: ListLike[Number] = DEFAULT_PRESSURE,
    troposphere: ListLike[Number] | None = None,
    # - dataloader
    batch_size: int = 1,
    timeout: float | None = None,
) -> DataLoader[Array[[N, N, N, N, N], np.float_]]:
    return DataLoader(
        consumer(
            dx,
            dy,
            *dsets,
            height=height,
            width=width,
            rate=rate,
            target_projection=target_projection,
            pressure=pressure,
            troposphere=troposphere,
            timeout=timeout,
        ),
        batch_size=batch_size,
        timeout=timeout or 0,
    )
