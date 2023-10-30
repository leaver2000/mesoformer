from typing import Callable, Iterable, TypeVar

import torch

_T = TypeVar("_T", str, int, float, bool, torch.Tensor)

def map_reduce(f: Callable[[_T, _T], _T]) -> Callable[[Iterable[_T], Iterable[_T]], Iterable[_T]]: ...
def truediv(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...
def floordiv(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...
def equals(x: Iterable[_T], y: Iterable[_T]) -> Iterable[bool]: ...
def mod(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...
def pow_(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...
def add(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...
def sub(x: Iterable[_T], y: Iterable[_T]) -> Iterable[_T]: ...