import metpy.calc
import metpy.units as munits
import numpy as np
from numpy.typing import NDArray

STANDARD_SURFACE_PRESSURE = 1_013.25

BASE_EXTENT = 200
URMA_SPACING = 2.5
DEFAULT_SCALE = 1.225
STANDARD_ATMOSPHERE = np.array([STANDARD_SURFACE_PRESSURE, 850.0, 700.0, 500.0, 300.0, 200.0])
X = metpy.calc.pressure_to_height_std(STANDARD_ATMOSPHERE * munits.units.hPa).m


log_p = X * np.log1p(STANDARD_ATMOSPHERE)


def get_spacing(scale=DEFAULT_SCALE, offset=URMA_SPACING) -> NDArray[np.float_]:
    """
    >>> resample.STANDARD_ATMOSPHERE
    array([1013.25,  850.  ,  700.  ,  500.  ,  300.  ,  200.  ])
    >>> resample.get_spacing()
    array([ 2.5       , 14.53714852, 26.66521388, 44.9283812 , 66.53269052,
        78.95511166])
    """
    return (log_p * scale) + offset
