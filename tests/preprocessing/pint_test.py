import src.mesoformer.preprocessing.pint as atmo

import metpy.calc as mpcalc
import numpy as np
import pytest


@pytest.mark.parametrize(
    "pressure",
    [
        [1013.25, 1000, 500] * atmo.hPa,
    ],
)
def tests_pressure2height(pressure) -> None:
    heights = atmo.pressure2height(pressure)
    assert np.allclose(atmo.height2pressure(heights).m, pressure.m)


@pytest.mark.parametrize(
    "heights",
    [
        [0, 1, 12] * atmo.km,
    ],
)
def tests_height2pressure(heights) -> None:
    pressure = atmo.height2pressure(heights)
    assert np.allclose(atmo.pressure2height(pressure).m, heights.m)
