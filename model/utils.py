# ruff: noqa: F401
from __future__ import annotations

from typing import TypeVar, TypeAlias, TYPE_CHECKING, Iterable, Any, Callable, ParamSpec
import itertools
import math
import torch
import operator

_T = TypeVar("_T", str, int, float, bool, torch.Tensor)

Single: TypeAlias = tuple[_T]
Pair: TypeAlias = tuple[_T, _T]
Triple: TypeAlias = tuple[_T, _T, _T]
Quadruple: TypeAlias = tuple[_T, _T, _T, _T]


def zip_reduce(f: Callable[[_T, _T], _T]) -> Callable[[Iterable[_T], Iterable[_T]], Iterable[_T]]:
    return lambda x, y: itertools.starmap(f, zip(x, y))


if TYPE_CHECKING:

    def single(x: _T | Single[_T]) -> _T:
        ...

    def pair(x: _T | Pair[_T]) -> Pair[_T]:
        ...

    def triple(x: _T | Triple[_T]) -> Triple[_T]:
        ...

    def quadruple(x: _T | Quadruple[_T]) -> Quadruple[_T]:
        ...

    def zip_truediv(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]:
        ...

    def zip_floordiv(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]:
        ...

    def zip_equals(x: Iterable[_T], y: Iterable[_T]) -> Iterable[bool]:
        ...

else:
    from torch.nn.modules.utils import (
        _single as single,
        _pair as pair,
        _triple as triple,
        _quadruple as quadruple,
    )

    zip_truediv = zip_reduce(operator.truediv)
    zip_floordiv = zip_reduce(operator.floordiv)
    zip_equals = zip_reduce(operator.eq)


def all_equals(x: Iterable[Any], y: Iterable[Any]) -> bool:
    return all(zip_equals(x, y))


def einsum_transpose(
    equation: str, unsqueeze: tuple[int, ...], output_shape: tuple[int, ...]
) -> Callable[[torch.Tensor], torch.Tensor]:
    x, y = equation.split("->")
    assert set(x) == set(y)
    assert math.prod(unsqueeze) == math.prod(output_shape)
    return lambda __x: torch.einsum(equation, __x.reshape(unsqueeze)).reshape(output_shape)
