from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

import torch

_P = ParamSpec("_P")
_T = TypeVar("_T")


class GenericModule(torch.nn.Module, Generic[_P, _T]):
    if TYPE_CHECKING:

        @abc.abstractmethod
        def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            ...

    @abc.abstractmethod
    def forward(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        ...
