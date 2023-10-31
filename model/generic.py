from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

import torch
from typing_extensions import Self

from .utils import DictStrAny

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


class IOModule(torch.nn.Module, Generic[_P, _T], abc.ABC):
    def to_disk(self, f: str) -> None:
        torch.save({"__constructor__": self._constructor_kwargs, "__state__": self.state_dict()}, f)

    @classmethod
    def from_disk(cls, f: str) -> Self:
        x = torch.load(f)
        m = cls(**x["__constructor__"])
        m.load_state_dict(x["__state__"])
        return m

    @property
    @abc.abstractmethod
    def _constructor_kwargs(self) -> DictStrAny:
        ...
