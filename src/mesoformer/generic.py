"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import queue
import random
import threading

import numpy as np
from torch.utils.data import IterableDataset

from .typing import (  # T,; P,
    Any,
    ArrayLike,
    Concatenate,
    DictStrAny,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    N,
    NDArray,
    NestedSequence,
    ParamSpec,
    Self,
    Sequence,
    T,
    TypeVar,
)
from .utils import indent_kv

K = TypeVar("K", bound=Hashable)
S = TypeVar("S")
P = ParamSpec("P")


class Data(Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> Iterable[tuple[str, T]]:
        ...

    def to_dict(self) -> dict[str, T]:
        return dict(self.data)

    def __repr__(self) -> str:
        data = indent_kv(*self.data)
        return "\n".join([f"{self.__class__.__name__}:"] + data)


class DataMapping(Mapping[K, T], Data[T]):
    def __init__(self, data: Mapping[K, T]) -> None:
        super().__init__()
        self._data: Final[Mapping[K, T]] = data

    @property
    def data(self) -> Iterable[tuple[K, T]]:
        yield from self.items()

    def __getitem__(self, key: K) -> T:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # def to_dict(self) -> dict[K, T]:
    #     return dict(self)


class DataWorker(Mapping[K, T]):
    __slots__ = ("_indices", "config")

    def __init__(self, *, indices: Iterable[K], **config: Any) -> None:
        super().__init__()
        self._indices: Final[list[K]] = list(indices)
        self.config: Final[DictStrAny] = config

    @property
    def indices(self) -> tuple[K, ...]:
        return tuple(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[K]:
        return iter(self._indices)

    def __repr__(self) -> str:
        x = self._indices
        return f"{self.__class__.__name__}(indices=[{x[0]} ... {x[-1]}])"

    def split(self, frac: float = 0.8) -> tuple[Self, Self]:
        cls = type(self)
        n = int(len(self) * frac)
        left, right = self._indices[:n], self._indices[n:]
        return cls(indices=left, **self.config), cls(indices=right, **self.config)

    def shuffle(self, *, seed: int) -> Self:
        random.seed(seed)
        random.shuffle(self._indices)
        return self

    @abc.abstractmethod
    def __getitem__(self, key: K) -> T:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...


class ABCDataConsumer(Generic[K, T], IterableDataset[T]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, worker: DataWorker[K, T], *, maxsize: int = 0, timeout: float | None = None) -> None:
        super().__init__()
        self.thread: Final = threading.Thread(target=self._target, name=self.name, daemon=True)
        self.queue: Final = queue.Queue[T](maxsize=maxsize)
        self.worker: Final = worker
        self.timeout: Final = timeout

    def _target(self) -> None:
        for index in self.worker.keys():
            self.queue.put(self.worker[index], block=True, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.worker)

    def __iter__(self) -> Iterator[T]:
        if not self.thread.is_alive():
            self.start()
        # range is the safest option here, because the queue size may change
        # during iteration, and a While loop is difficult to break out of.
        return (self.queue.get(block=True, timeout=self.timeout) for _ in range(len(self)))

    def start(self):
        self.worker.start()
        self.thread.start()
        return self


class ABCArray(Sequence[T], Generic[S, T], abc.ABC):
    ...


class Array(ABCArray[Concatenate[P], T]):
    def __init__(self, data: NestedSequence[T] | NDArray[T]) -> None:
        super().__init__()
        self._data = np.asanyarray(data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:\n{np.array2string(self._data)}"

    def __array__(self) -> NDArray[T]:
        return self._data

    def __getitem__(self, idx: N | tuple[N, ...] | Array[..., np.bool_]) -> Array[..., T]:
        return Array(self._data[idx])

    @property
    def size(self) -> int:
        return self._data.size

    def is_in(self, x: ArrayLike) -> Array[P, np.bool_]:
        return Array(np.isin(self._data, x))  # type: ignore

    def to_numpy(self) -> NDArray[T]:
        return self._data

    def item(self) -> T:
        return self._data.item()
