"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

from typing import Any, MutableMapping

import pyproj

from .enum_table import TableEnum, auto_field


class IndependentVariables(str, TableEnum):
    metadata: MutableMapping[str, Any]  # type: ignore

    @staticmethod
    def _generate_next_value_(name: str, *_):
        return name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


def get_crs(name: str) -> pyproj.CRS:
    # TODO: move this to disk
    if name == ERA5.__name__:
        cf = {
            "crs_wkt": 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
            "geographic_crs_name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "World Geodetic System 1984 ensemble",
            "grid_mapping_name": "latitude_longitude",
        }
    elif name == URMA.__name__:
        cf = {
            "geographic_crs_name": "NDFD CONUS 2.5km Lambert Conformal Conic",
            "projected_crs_name": "NDFD",
            "grid_mapping_name": "lambert_conformal_conic",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31424518,
            "inverse_flattening": 298.25722356301,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "WGS84",
            "latitude_of_projection_origin": 20.191999,
            "longitude_of_projection_origin": 238.445999,
            "standard_parallel": 25,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "units": "m",
        }
    else:
        raise ValueError(f"Unknown CRS {name!r}")
    return pyproj.CRS.from_cf(cf)


class DependentVariables(IndependentVariables):
    @classmethod  # type:ignore
    @property
    def crs(cls) -> pyproj.CRS:
        md = cls.metadata
        if "crs" not in md:
            md["crs"] = get_crs(cls.__name__)

        return md["crs"]


# =====================================================================================================================


class Dimensions(IndependentVariables):
    T = auto_field(aliases=["t", "time"])
    Z = auto_field(aliases=["z", "level", "height", "altitude"])
    Y = auto_field(aliases=["y", "latitude", "grid_latitude"])
    X = auto_field(aliases=["x", "longitude", "grid_longitude"])


DIMENSIONS = T, Z, Y, X = (
    Dimensions.T,
    Dimensions.Z,
    Dimensions.Y,
    Dimensions.X,
)


class Coordinates(IndependentVariables):
    time = auto_field(axis=(T,))
    vertical = auto_field(aliases=["level", "height", "altitude"], axis=(Z,))
    latitude = auto_field(aliases=["grid_latitude"], axis=(Y, X))
    longitude = auto_field(aliases=["grid_longitude"], axis=(Y, X))

    @property
    def axis(self) -> tuple[Dimensions, ...]:
        return self.metadata["axis"]


COORDINATES = TIME, LVL, LAT, LON = (
    Coordinates.time,
    Coordinates.vertical,
    Coordinates.latitude,
    Coordinates.longitude,
)


class ERA5(DependentVariables):
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

    Z = auto_field("geopotential", aliases=["z"], units="m**2 s**-2")
    Q = auto_field("specific_humidity", aliases=["q"], units="kg kg**-1")
    T = auto_field("temperature", aliases=["t"], units="K")
    U = auto_field("u_component_of_wind", aliases=["u"], units="m s**-1")
    V = auto_field("v_component_of_wind", aliases=["v"], units="m s**-1")
    W = auto_field("vertical_velocity", aliases=["w"], units="Pa s**-1")


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


class URMA(DependentVariables):
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
