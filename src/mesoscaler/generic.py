"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import queue
import random
import threading

from ._torch_data import IterableDataset  # type: ignore

from .typing import (
    Any,
    AnyArrayLike,
    Callable,
    DictStrAny,
    Final,
    Generic,
    Hashable,
    HashableT,
    Iterable,
    Iterator,
    Mapping,
    Self,
    T,
    TypeVar,
    _T_co,
    overload,
)
from .utils import is_array_like, join_kv

K = TypeVar("K")
R = TypeVar("R")
ListIndex = list[int] | list[bool] | list[str]


class Loc(Generic[K, R]):
    """
    ```
    s = pd.Series({"a": 1, "b": 2, "c": 3})  # type: pd.Series[int]
    loc = Loc(functools.partial(np.asarray, dtype=np.float_), s)  # type: Loc[float, NDArray[np.float_]]
    a = loc["a"]  # type: int
    bc = loc[["b", "c"]]  # type: NDArray[np.float_]
    arr = np.array([1, 2, 3]).astype(np.float_)  # type: NDArray[np.float_]
    list_loc = Loc(list, arr)  # type: Loc[np.float_, list[np.float_]]
    z = list_loc[0]  # type: np.float_
    assert isinstance(z, np.floating)
    x = list_loc[:]  # type: list[np.float_]
    ```
    """

    def __init__(self, hook: Callable[[AnyArrayLike[_T_co]], R], x: AnyArrayLike[_T_co]) -> None:
        self._data = x
        self._hook = hook

    @overload
    def __getitem__(self, item: K) -> _T_co:
        ...

    @overload
    def __getitem__(self, item: list[K | bool | Any]) -> R:
        ...

    def __getitem__(self, item: list[K | bool | Any]) -> _T_co | R:
        x = self._data[item]  # type: ignore
        if is_array_like(x):
            return self._hook(x)
        return x  # type: ignore


class Data(Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> Iterable[tuple[Hashable, T]]:
        ...

    def to_dict(self) -> dict[Hashable, T]:
        return dict(self.data)

    def __repr__(self) -> str:
        return join_kv(self.__class__.__name__, *self.data)


class DataMapping(Mapping[HashableT, T], Data[T]):
    def __init__(self, data: Mapping[HashableT, T]) -> None:
        super().__init__()
        self._data: Final[Mapping[HashableT, T]] = data

    @property
    def data(self) -> Iterable[tuple[HashableT, T]]:
        yield from self.items()  # type: ignore

    def __getitem__(self, key: HashableT) -> T:
        return self._data[key]

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class DataWorker(Mapping[HashableT, T], Data[T]):
    __slots__ = ("_indices", "config")

    def __init__(self, *, indices: Iterable[HashableT], **config: Any) -> None:
        super().__init__()
        self._indices: Final[list[HashableT]] = list(indices)
        self.config: Final[DictStrAny] = config

    @property
    def indices(self) -> tuple[HashableT, ...]:
        return tuple(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[HashableT]:
        return iter(self._indices)

    @property
    def data(self):
        x = self._indices
        return [
            ("indices", f"[{x[0]} ... {x[-1]}]"),
        ]

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
    def __getitem__(self, key: HashableT) -> T:
        ...

    @abc.abstractmethod
    def start(self) -> None:
        ...


class ABCDataConsumer(Generic[HashableT, T], IterableDataset[T]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, worker: DataWorker[HashableT, T], *, maxsize: int = 0, timeout: float | None = None) -> None:
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
