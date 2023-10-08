"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

from .enum_table import TableEnum, auto_field as field


class Table(str, TableEnum):
    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


class Dimensions(Table):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        return name.upper()

    T = field(aliases=["time"])
    Z = field(aliases=["level", "height", "altitude"])
    Y = field(aliases=["latitude", "grid_latitude"])
    X = field(aliases=["longitude", "grid_longitude"])


DIMENSIONS = T, Z, Y, X = (
    Dimensions.T,
    Dimensions.Z,
    Dimensions.Y,
    Dimensions.X,
)


class Coordinates(Table):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        return name.lower()

    TIME = field(aliases=["time"], metadata={"axis": (T,)})
    VERTICAL = field(aliases=["level", "height", "altitude"], metadata={"axis": (Z,)})
    LATITUDE = field(aliases=["latitude", "grid_latitude"], metadata={"axis": (Y, X)})
    LONGITUDE = field(aliases=["longitude", "grid_longitude"], metadata={"axis": (Y, X)})

    @property
    def axis(self) -> tuple[Dimensions, ...]:
        return self.metadata["axis"]


COORDINATES = TIME, LVL, LAT, LON = (
    Coordinates.TIME,
    Coordinates.VERTICAL,
    Coordinates.LATITUDE,
    Coordinates.LONGITUDE,
)


class ERA5(Table):
    r"""
    | member_name   | short_name   | standard_name       | long_name           | type_of_level   | units      |
    |:--------------|:-------------|:--------------------|:--------------------|:----------------|:-----------|
    | Z             | z            | geopotential        | Geopotential        | isobaricInhPa   | m**2 s**-2 |
    | Q             | q            | specific_humidity   | Specific humidity   | isobaricInhPa   | kg kg**-1  |
    | T             | t            | temperature         | Temperature         | isobaricInhPa   | K          |
    | U             | u            | u_component_of_wind | U component of wind | isobaricInhPa   | m s**-1    |
    | V             | v            | v_component_of_wind | V component of wind | isobaricInhPa   | m s**-1    |
    | W             | w            | vertical_velocity   | Vertical velocity   | isobaricInhPa   | Pa s**-1   |
    """

    Z = field("geopotential", aliases=["z"], metadata={"units": "m**2 s**-2"})
    Q = field("specific_humidity", aliases=["q"], metadata={"units": "kg kg**-1"})
    T = field("temperature", aliases=["t"], metadata={"units": "K"})
    U = field("u_component_of_wind", aliases=["u"], metadata={"units": "m s**-1"})
    V = field("v_component_of_wind", aliases=["v"], metadata={"units": "m s**-1"})
    W = field("vertical_velocity", aliases=["w"], metadata={"units": "Pa s**-1"})


ERA5_VARS = (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    TEMPERATURE,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
    VERTICAL_VELOCITY,
) = (
    ERA5.Z,
    ERA5.Q,
    ERA5.T,
    ERA5.U,
    ERA5.V,
    ERA5.W,
)


class URMA(Table):
    """
    | member_name   | short_name   | standard_name           | long_name                    | type_of_level         | units       |
    |:--------------|:-------------|:------------------------|:-----------------------------|:----------------------|:------------|
    | CEIL          | ceil         | ceiling                 | cloud ceiling                | cloudCeiling          | m           |
    | D2M           | d2m          | dewpoint_temperature_2m | 2 meter dewpoint temperature | heightAboveGround     | K           |
    | SH2           | sh2          | specific_humidity_2m    | 2 meter specific humidity    | heightAboveGround     | kg kg**-1   |
    | SP            | sp           | surface_pressure        | surface pressure             | surface               | Pa          |
    | T2M           | t2m          | temperature_2m          | 2 meter temperature          | heightAboveGround     | K           |
    | TCC           | tcc          | total_cloud_cover       | total cloud cover            | atmosphereSingleLayer | %           |
    | U10           | u10          | u_wind_component_10m    | 10 meter u wind component    | heightAboveGround     | m s**-1     |
    | V10           | v10          | v_wind_component_10m    | 10 meter v wind component    | heightAboveGround     | m s**-1     |
    | VIS           | vis          | visibility              | visibility                   | surface               | m           |
    | WDIR10        | wdir10       | wind_direction_10m      | 10 meter wind direction      | heightAboveGround     | Degree true |
    | SI10          | si10         | wind_speed_10m          | 10 meter wind speed          | heightAboveGround     | m s**-1     |
    | GUST          | gust         | wind_speed_gust         | wind speed gust              | heightAboveGround     | m s**-1     |
    | OROG          | orog         | orography               | surface orography            | surface               | m           |
    """

    TCC = "total_cloud_cover"
    CEIL = "ceiling"
    U10 = "u_wind_component_10m"
    V10 = "v_wind_component_10m"
    SI10 = "wind_speed_10m"
    GUST = "wind_speed_gust"
    WDIR10 = "wind_direction_10m"
    T2M = "temperature_2m"
    D2M = "dewpoint_temperature_2m"
    SH2 = "specific_humidity_2m"
    SP = "surface_pressure"
    VIS = "visibility"
    OROG = "orography"


URMA_VARS = (
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
) = (
    URMA.TCC,
    URMA.CEIL,
    URMA.U10,
    URMA.V10,
    URMA.SI10,
    URMA.GUST,
    URMA.WDIR10,
    URMA.T2M,
    URMA.D2M,
    URMA.SH2,
    URMA.SP,
    URMA.VIS,
    URMA.OROG,
)


def main() -> None:
    assert ERA5.Z == "geopotential"
    assert ERA5("z") == "geopotential"

    assert ERA5(["z"]) == ["geopotential"]
    assert ERA5(["z", "q"]) == ["geopotential", "specific_humidity"]
    assert ERA5.to_list() == [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]
    try:
        ERA5("nope")
        raise AssertionError
    except ValueError:
        pass
    assert URMA[["TCC"]] == ["total_cloud_cover"]
    print(Dimensions.to_frame())
    assert LAT.axis == (Y, X)


if __name__ == "__main__":
    main()
