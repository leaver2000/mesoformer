from __future__ import annotations

__all__ = [
    "normalize",
    "normalized_scale",
    "sort_unique",
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
import itertools
import json
import textwrap
import types
import urllib.parse
from collections.abc import Sequence
import functools
import numpy as np
import toml
import torch
from frozenlist import FrozenList
from scipy.interpolate import RegularGridInterpolator

try:
    get_ipython  # type: ignore
    import tqdm.notebook as tqdm
except NameError:
    import tqdm

from .typing import (
    Any,
    Array,
    Callable,
    Iterable,
    Iterator,
    ListLike,
    Mapping,
    N,
    NDArray,
    Pair,
    Sequence,
    StrPath,
    T,
    TypeVar,
)

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

TensorT = TypeVar("TensorT", torch.Tensor, Array[..., Any])


# =====================================================================================================================
# - array/tensor utils
# =====================================================================================================================
def normalize(x: TensorT) -> TensorT:
    """
    Normalize the input tensor along the specified dimensions.

    Args:
        x: Input tensor to be normalized.
        **kwargs: Additional arguments to be passed to the `min` and `max` functions.

    Returns:
        Normalized tensor.

    Raises:
        TypeError: If the input tensor is not a numpy array or a PyTorch tensor.
    """
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Input tensor must be a numpy array or a PyTorch tensor.")
    return (x - x.min()) / (x.max() - x.min())  # type: ignore


def normalized_scale(x: TensorT, rate: float = 1.0) -> TensorT:
    """
    Scales the input tensor `x` by a factor of `rate` after normalizing it.

    Args:
        x (numpy.ndarray or torch.Tensor): The input tensor to be normalized and scaled.
        rate (float, optional): The scaling factor. Defaults to 1.0.

    Returns:
        numpy.ndarray or torch.Tensor: The normalized and scaled tensor.
    """
    x = normalize(x)
    x *= rate
    x += 1

    return x


def log_scale(x: NDArray[np.number], rate: float = 1.0) -> NDArray[np.float_]:
    return normalized_scale(np.log(x), rate=rate)


def sort_unique(x: ListLike[T]) -> NDArray[T]:
    """
    Sorts the elements of the input array `x` in ascending order and removes any duplicates.

    Parameters
    ----------
    x : ListLike[T]
        The input array to be sorted and made unique.

    Returns
    -------
    NDArray[T]
        A new array containing the sorted, unique elements of `x`.

    Examples
    --------
    >>> sort_unique([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    array([1, 2, 3, 4, 5, 6, 9])
    """
    return np.sort(np.unique(np.asanyarray(x)))


def square_space(in_size: int, out_size: int) -> tuple[Pair[Array[[N], Any]], Pair[Array[[N, N], Any]]]:
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
    arr: Array[[N, N, ...], T],
    *,
    img_size: int = 256,
    method: str = "linear",
) -> Array[Nd[N, N, ...], T]:  # type: ignore
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
        return arr  # type: ignore
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
    # - repr utils
    # =====================================================================================================================


_array2string = functools.partial(
    np.array2string,
    max_line_width=72,
    precision=2,
    separator=" ",
    floatmode="fixed",
)


def indent_pair(k: str, v: Any, l_pad: int, prefix="  ") -> str:
    if isinstance(v, np.ndarray):
        v = _array2string(v)
    return textwrap.indent(f"{k.rjust(l_pad)}: {v}", prefix=prefix)


def indent_kv(*args: tuple[str, Any], prefix="  ") -> list[str]:
    l_pad = max(len(k) for k, _ in args)

    return [indent_pair(k, v, l_pad, prefix) for k, v in args]


# =====================================================================================================================
# - list utils
# =====================================================================================================================
def frozen_list(item: Sequence[_T1]) -> list[_T1]:
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


# =====================================================================================================================
# - mapping utils
# =====================================================================================================================
def nested_proxy(data: Mapping[str, Any]) -> types.MappingProxyType[str, Any]:
    return types.MappingProxyType({k: nested_proxy(v) if isinstance(v, Mapping) else v for k, v in data.items()})


# =====================================================================================================================
# - iterable utils
# =====================================================================================================================
def better_iter(x: _T1 | Iterable[_T1]) -> Iterator[_T1]:
    """

    >>> list(better_iter((1,2,3,4)))
    [1, 2, 3, 4]
    >>> list(better_iter('hello'))
    ['hello']
    >>> list(better_iter(['hello', 'world']))
    ['hello', 'world']
    """
    if not isinstance(x, Iterable):
        raise TypeError(f"expected an iterable, but got {type(x)}")
    return iter([x] if isinstance(x, str) else x)  # type: ignore


def find(func: Callable[[_T1], bool], x: Iterable[_T1]) -> _T1:
    try:
        return next(filter(func, x))
    except StopIteration as e:
        raise ValueError(f"no element in {x} satisfies {func}") from e


def squish_map(func: Callable[[_T1], _T2], __iterable: _T1 | Iterable[_T1], *args: _T1) -> map[_T2]:
    """
    >>> assert list(squish_map(lambda x: x, "foo", "bar", "baz")) == ["foo", "bar", "baz"]
    >>> assert list(squish_map(str, range(3), 4, 5)) == ["0", "1", "2", "4", "5"]
    >>> assert list(squish_map("hello {}".format, (x for x in ("foo", "bar")), "spam")) == ["hello foo", "hello bar", "hello spam"]
    """

    return map(func, itertools.chain(better_iter(__iterable), iter(args)))


# =====================================================================================================================
# - IO utils
# =====================================================================================================================
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
