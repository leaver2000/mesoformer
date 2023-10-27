import sys
import argparse
from mesoscaler.core import ArrayWorker, Mesoscale, DependentDataset
from mesoscaler.generic import DataConsumer
from mesoscaler.enums import (
    URMA,
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
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    VERTICAL_VELOCITY,
)
from torch.utils.data import DataLoader

from . import data
import functools


configs = {
    "default": [
        data.Config(
            store="/mnt/data/urma2p5.zarr",
            depends=[
                SURFACE_PRESSURE,
                TEMPERATURE_2M,
                SPECIFIC_HUMIDITY_2M,
                U_WIND_COMPONENT_10M,
                V_WIND_COMPONENT_10M,
            ],
        ),
        data.Config(
            store="/mnt/data/era5/",
            depends=[GEOPOTENTIAL, TEMPERATURE, SPECIFIC_HUMIDITY, U_COMPONENT_OF_WIND, V_COMPONENT_OF_WIND],
        ),
    ],
    "wind": [
        data.Config(
            store="/mnt/data/urma2p5.zarr",
            depends=[U_WIND_COMPONENT_10M, V_WIND_COMPONENT_10M],
        ),
        data.Config(
            store="/mnt/data/era5/",
            depends=[U_COMPONENT_OF_WIND, V_COMPONENT_OF_WIND],
        ),
    ],
}


def main(
    loader: data.DataLoader,
) -> int:
    ...
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=int, default=200)
    parser.add_argument("--dy", type=int, default=200)
    # data configs
    parser.add_argument("--config", type=str, default="wind")
    args = parser.parse_args()
    dsets = [DependentDataset.from_zarr(**cfg) for cfg in configs[args.config]]

    # parser.add_argument("--height", type=int, default=80)
