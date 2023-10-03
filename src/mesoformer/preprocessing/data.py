import numpy as np

from ..typing import Any, NDArray, ParamSpec, TypeVar
from ..utils import as_any_array

FloatingDTypeT = TypeVar("FloatingDTypeT", bound=np.floating, covariant=True)
AnyT_co = TypeVar("AnyT_co", bound=Any, covariant=True)

P = ParamSpec("P")


@as_any_array(np.float32)
def low_high_scaler(x: NDArray[np.number], l: float = 0, h: float = 1.2, *, axis=None) -> NDArray[np.float32]:
    return (x - x.min(axis)) / (x.max(axis) - x.min(axis)) * (h - l) + l


@as_any_array(np.float32)
def min_max_scaler(x: NDArray[np.number], *, axis=None) -> NDArray[np.float32]:
    return (x - x.min(axis)) / (x.max(axis) - x.min(axis))


@as_any_array(np.float32)
def std_scaler(x: NDArray[np.number], *, axis=None) -> NDArray[np.float32]:
    return (x - x.mean(axis)) / x.std(axis)


@as_any_array(np.float32)
def max_abs_scaler(x: NDArray[np.number], *, axis=None) -> NDArray[np.float32]:
    return x / np.abs(x).max(axis)


@as_any_array(np.float32)
def normalize(x: NDArray[np.number]) -> NDArray[np.float32]:
    return (x - x.min(keepdims=True)) / (x.max(keepdims=True) - x.min(keepdims=True))
