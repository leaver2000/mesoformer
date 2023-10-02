from __future__ import annotations

__all__ = [
    "arange_slice",
    "frozen_list",
    "interp_frames",
    "square_space",
    "url_join",
    "dump_json",
    "dump_jsonl",
    "dump_toml",
    "iter_jsonl",
    "load_json",
    "load_toml",
    "tqdm",
]
import json
import textwrap
import types
import urllib.parse

import numpy as np
import toml
from frozenlist import FrozenList
from scipy.interpolate import RegularGridInterpolator
from typing import Concatenate

try:
    get_ipython  # type: ignore
    import tqdm.notebook as tqdm
except NameError:
    import tqdm

import itertools
import functools
from .typing import (
    Any,
    ArrayLike,
    AnyT,
    Array,
    Callable,
    Iterable,
    Mapping,
    N,
    Nd,
    Pair,
    ParamSpec,
    Sequence,
    StrPath,
    TypeVar,
    NDArray,
)

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T = TypeVar("T")
P = ParamSpec("P")
FloatingDTypeT = TypeVar("FloatingDTypeT", bound=np.floating, covariant=True)
AnyT_co = TypeVar("AnyT_co", bound=Any, covariant=True)
R_co = TypeVar("R_co", bound=Callable)
P = ParamSpec("P")


def as_any_array(dtype: type[FloatingDTypeT]):
    def decorator(
        func: Callable[Concatenate[NDArray[np.number], P], AnyT_co]
    ) -> Callable[Concatenate[ArrayLike, P], AnyT_co]:
        @functools.wraps(func)
        def wrapper(x: ArrayLike, *args: P.args, **kwargs: P.kwargs) -> AnyT_co:
            return func(np.asanyarray(x, dtype=dtype), *args, **kwargs)

        return wrapper

    return decorator


def iter_not_strings(x: T1 | Iterable[T1]) -> Iterable[T1]:
    """
    >>> list(iter_not_strings(None))
    [None]
    >>> list(iter_not_strings((1,2,3,4)))
    [1, 2, 3, 4]
    >>> list(iter_not_strings('hello'))
    ['hello']
    >>> list(iter_not_strings(['hello', 'world']))
    ['hello', 'world']
    """
    return iter([x] if isinstance(x, str) or not isinstance(x, Iterable) else x)  # type: ignore


def squish_map(func: Callable[[T1], T2], __iterable: T1 | Iterable[T1], *args: T1) -> map[T2]:
    """
    >>> assert list(squish_map(lambda x: x, "foo", "bar", "baz")) == ["foo", "bar", "baz"]
    >>> assert list(squish_map(str, range(3), 4, 5)) == ["0", "1", "2", "4", "5"]
    >>> assert list(squish_map("hello {}".format, (x for x in ("foo", "bar")), "spam")) == ["hello foo", "hello bar", "hello spam"]
    """

    return map(func, itertools.chain(iter_not_strings(__iterable), iter(args)))


# =====================================================================================================================
def frozen_list(item: Sequence[T]) -> list[T]:
    fl = FrozenList(item)
    fl.freeze()
    return fl  # type: ignore


def arange_slice(
    start: int, stop: int | None, rows: int | None, ppad: int | None, step: int | None = None
) -> list[slice]:
    if stop is None:
        start, stop = 0, start
    if ppad == 0:
        ppad = None

    elif stop < start:
        raise ValueError(f"stop ({stop}) must be less than start ({start})")

    stop += 1  # stop is exclusive

    if rows is None:
        rows = 1

    if ppad is not None:
        if ppad > rows:
            raise ValueError(f"pad ({ppad}) must be less than freq ({rows})")
        it = zip(range(start, stop, rows // ppad), range(start + rows, stop, rows // ppad))
    else:
        it = zip(range(start, stop, rows), range(start + rows, stop, rows))
    return [np.s_[i:j:step] for i, j in it if j <= stop]


def square_space(in_size: int, out_size: int) -> tuple[Pair[Array[Nd[N], Any]], Pair[Array[Nd[N, N], Any]]]:
    """
    >>> points, values = squarespace(4, 6)
    >>> points
    (array([0.        , 0.08333333, 0.16666667, 0.25      ]), array([0.        , 0.08333333, 0.16666667, 0.25      ]))
    >>> grid
    (array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
           [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ],
           [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
           [0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ],
           [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]), array([[0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25]]))
    """
    xy1 = np.linspace(0, 1.0 / in_size, in_size)
    xy2 = np.linspace(0, 1.0 / in_size, out_size)
    return (xy1, xy1), tuple(np.meshgrid(xy2, xy2, indexing="ij"))  # type: ignore[return-value]


def interp_frames(
    arr: Array[Nd[N, N, ...], AnyT],  # type: ignore
    *,
    img_size: int = 256,
    method: str = "linear",
) -> Array[Nd[N, N, ...], AnyT]:  # type: ignore
    """
    Interpolate the first two equally shaped dimensions of an array to the new `patch_size`.
    using `scipy.interpolate.RegularGridInterpolator`.

    >>> import numpy as np
    >>> import atmoformer.utils
    >>> arr = np.random.randint(0, 255, (384, 384, 49))
    >>> atmoformer.utils.interpatch(arr, 768).shape
    (768, 768, 49)
    """
    x, y = arr.shape[:2]
    if x != y:  # first two dimensions must be equal
        raise ValueError(f"array must be square, but got shape: {arr.shape}")
    elif x == img_size == y:  # no interpolation needed
        return arr
    points, values = square_space(x, img_size)
    interp = RegularGridInterpolator(points, arr, method=method)
    return interp(values).astype(arr.dtype)


def url_join(url: str, *args: str, allow_fragments: bool = True) -> str:
    """
    >>> from metformer.utils import url_join
    >>> url_join('https://example.com', 'images', 'cats')
    'https://example.com/images/cats'
    >>> url_join('https://example.com', '/images')
    'https://example.com/images'
    >>> url_join('https://example.com/', 'images')
    'https://example.com/images'
    """
    if not url.startswith("http"):
        raise ValueError(f"invalid url: {url}")
    return urllib.parse.urljoin(url, "/".join(x.strip("/") for x in args), allow_fragments=allow_fragments)


# =====================================================================================================================
# - basic IO functions
def dump_toml(obj: Any, src: StrPath, preserve=True, numpy: bool = False) -> None:
    with open(src, "w") as f:
        toml.dump(
            obj, f, encoder=toml.TomlNumpyEncoder(preserve=preserve) if numpy else toml.TomlEncoder(preserve=preserve)
        )


def load_toml(src: StrPath) -> Any:
    with open(src, "r") as f:
        return toml.load(f)


def dump_json(obj: Any, src: StrPath) -> None:
    with open(src, "w") as f:
        json.dump(obj, f)


def load_json(src: StrPath) -> Any:
    with open(src, "r") as f:
        return json.load(f)


def dump_jsonl(obj: Iterable[Any], src: StrPath) -> None:
    with open(src, "w") as f:
        for x in obj:
            json.dump(x, f)
            f.write("\n")


def iter_jsonl(src: StrPath) -> Iterable[Any]:
    with open(src, "r") as f:
        for line in f:
            yield json.loads(line)


def indent_kv(*args: tuple[str, Any], prefix="  ") -> list[str]:
    return [textwrap.indent(f"{k}: {v}", prefix=prefix) for k, v in args]


def nested_proxy(data: Mapping) -> types.MappingProxyType[str, Any]:
    return types.MappingProxyType({k: nested_proxy(v) if isinstance(v, Mapping) else v for k, v in data.items()})
