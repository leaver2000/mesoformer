import sys

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

from . import main

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


if __name__ == "__main__":
    sys.exit(main.main())
#     dataset_sequence = ms.open_datasets([("data/urma.zarr", urma_dvars), ("data/era5.zarr", era5_dvars)])
#     sys.exit(main(dataset_sequence))
