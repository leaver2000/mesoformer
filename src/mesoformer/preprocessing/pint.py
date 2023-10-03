from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import *  # pyright: ignore

import numpy as np
import pint.registry
from numpy.typing import NDArray as _NDArray
from pint.facets.plain import PlainQuantity, PlainUnit
from pint.registry import Quantity, Unit

if TYPE_CHECKING:
    from ._literal_unit import LiteralUnit  # pyright: ignore
else:
    LiteralUnit = str

_T = TypeVar("_T", bound=Any)
NDArray: TypeAlias = _NDArray[_T]
NestedSequence: TypeAlias = "Sequence[_T | NestedSequence[_T]]"

Scalar: TypeAlias = float | int | Decimal | Fraction | np.number[Any]
ScalarT = TypeVar("ScalarT", bound=Scalar)


ArrayLike: TypeAlias = "NDArray[ScalarT] | NestedSequence[ScalarT]"

# ================================================================================
ureg: Final = pint.UnitRegistry()
ureg.default_format = ".3f~P"
unit: Final = ureg.Unit

# • [time]
s: Final = unit("second")
min_: Final = unit("minute")
hr: Final = unit("hour")

# • [temperature]
K: Final = unit("kelvin")
C: Final = unit("celsius")
F: Final = unit("fahrenheit")

# • [length]
# - metric
m = unit("meter")
km = unit("kilometer")
# - imperial

in_ = unit("inch")
ft = unit("foot")
mi = unit("mile")

# • [pressure]
Pa = unit("pascal")
hPa = unit("hectopascal")
kPa = unit("kilopascal")
mbar = unit("millibar")

# • energy
J = unit("joule")
kJ = unit("kilojoule")
cal = unit("calorie")
kcal = unit("kilocalorie")
mol = unit("mole")

# • [mass]
kg = unit("kilogram")

# • angle
deg = unit("degree")
rad = unit("radian")

# • speed
mps = m / s
kph = km / hr
mph = mi / hr
kts = unit("knot")

# • image processing
px = unit("pixel")
dpi = unit("dot/inch")
ppi = unit("pixel/inch")
dimensionless = unit("dimensionless")


# ================================================================================
@ureg.wraps(dimensionless, hPa)
def log_p(x) -> Quantity:
    return np.log(x)


def isscaler(x: Any) -> TypeGuard[Scalar]:
    return np.isscalar(x.magnitude if isinstance(x, Quantity) else x)


@overload
def quantity(
    x: ScalarT,
    unit: pint.Unit | Quantity | LiteralUnit | PlainUnit | PlainQuantity = dimensionless,
) -> PlainQuantity[ScalarT]:
    ...


@overload
def quantity(
    x: ArrayLike[ScalarT],
    unit: pint.Unit | Quantity | LiteralUnit | PlainUnit | PlainQuantity = dimensionless,
) -> PlainQuantity[NDArray[ScalarT]]:  # type: ignore
    ...


def quantity(
    x: ScalarT | ArrayLike[ScalarT],
    unit: pint.Unit | Quantity | LiteralUnit | PlainUnit | PlainQuantity = dimensionless,
) -> PlainQuantity[ScalarT] | PlainQuantity[NDArray[ScalarT]]:  # type: ignore
    unit = ureg(unit) if isinstance(unit, str) else unit
    return (x if isscaler(x) else np.asanyarray(x)) * unit  # type: ignore


# ================================================================================
EARTH_GRAVITY = g = quantity(9.80665, m / s**2)
"""
The standard acceleration due to gravity `g` at the Earth's surface.
"""
GRAVITATIONAL_CONSTANT = G = quantity(6.67408e-11, m**3 / kg / s**2)
"""
The gravitational constant `G` is a key quantity in Newton's law of universal gravitation.
"""
EARTH_RADIUS = Re = quantity(6371008.7714, m)
GEOCENTRIC_GRAVITATIONAL_CONSTANT = GM = quantity(3986005e8, m**3 / s**2)

EARTH_MASS = Me = GEOCENTRIC_GRAVITATIONAL_CONSTANT / GRAVITATIONAL_CONSTANT
ABSOLUTE_ZERO = K0 = quantity(-273.15, K)
MOLAR_GAS_CONSTANT = R = quantity(8.314462618, J / mol / K)


STANDARD_TEMPERATURE = t0 = quantity(288.0, K)
"""Standard temperature at sea level."""
STANDARD_PRESSURE = p0 = quantity(1013.25, hPa)
"""Standard pressure at sea level."""
STANDARD_LAPSE_RATE = gamma = quantity(6.5, K / km)
"""Standard lapse rate."""
STANDARD_PRESSURE_LEVELS = quantity([p0.m, *map(float, range(1000, 25 - 11, -25))], unit=hPa)

MOLAR_GAS_CONSTANT = R = quantity(8.314462618, J / mol / K)
DRY_AIR_MOLECULAR_WEIGHT_RATIO = Md = quantity(28.96546e-3, kg / mol)


# • Dry air
DRY_AIR_GAS_CONST = Rd = quantity(R / Md)
DRY_AIR_SPECIFIC_HEAT_RATIO = dash_r = quantity(1.4)
DRY_AIR_SPECIFIC_HEAT_PRESSURE = Cp_d = quantity(dash_r * Rd / (dash_r - 1))
DRY_AIR_SPECIFIC_HEAT_VOLUME = Cv_d = quantity(Cp_d / dash_r)
pot_temp_ref_press = P0 = 1000.0 * mbar
DRY_AIR_DENSITY_STP = rho_d = quantity(P0 / (Rd * -K0)).to(kg / m**3)
DRY_AIR_MOLECULAR_WEIGHT_RATIO = Md = 28.96546e-3 * kg / mol


def pressure2height(pressure: PlainQuantity[NDArray[Any]]) -> PlainQuantity[float]:
    return t0 / gamma * (1 - (pressure / p0) ** (Rd * gamma / g))


def height2pressure(height: PlainQuantity[float | int] | NDArray) -> PlainQuantity[float]:
    return p0 * (1 - (gamma / t0) * height) ** (g / (Rd * gamma))
