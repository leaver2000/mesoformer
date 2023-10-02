from __future__ import annotations

import itertools
import os
import types
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps  # type: ignore
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

if TYPE_CHECKING:
    from matplotlib.cm import ColormapRegistry

    colormaps: ColormapRegistry
    normalizations: types.MappingProxyType[str, Normalize]

from .typing import AnyArray, Batch, Channel, DictStrAny, Length, Time, Width, cast
from .utils import STATIC_FILES, load_json


# =====================================================================================================================
def __register_colors_and_norms() -> None:
    norms = {}
    for cfg in cast(list[DictStrAny], load_json(os.path.join(STATIC_FILES, "cmaps.json"))):
        # - create new cmap
        if isinstance(cfg["colors"], str):
            cmap = colormaps[cfg["colors"]]
        else:
            cmap = ListedColormap(cfg["colors"], name=cfg["channel"])  # type: ignore
            if bad := cfg.get("bad"):
                cmap.set_bad(bad)
            if under := cfg.get("under"):
                cmap.set_under(under)
            if over := cfg.get("over"):
                cmap.set_over(over)

        if cfg["channel"] not in colormaps:
            colormaps.register(cmap=cmap, name=cfg["channel"])

        # - create associated norm
        bounds = cfg["norm"]
        if isinstance(bounds, dict):
            norm = Normalize(**bounds)
        else:
            arr = np.array(bounds, dtype=np.float_)
            norm = BoundaryNorm(arr, ncolors=len(arr))
        norms[cfg["channel"]] = norm

    # - set the global normalizations mapping
    global normalizations
    normalizations = types.MappingProxyType(norms)


# rather than performing the registration at the global scope
# it is done in a function so the namespace is not polluted
# and ref-counts are not increased
__register_colors_and_norms()
# - remove the function from the namespace
del __register_colors_and_norms


def get_imconfig(channel: str) -> DictStrAny:
    return {"cmap": colormaps[channel], "norm": normalizations[channel]}


class AutoAnimator(FuncAnimation):
    """Plots a 4D or 5D tensor as an animation
    ```
    anmi = AutoAnimator(
        np.stack(samples)[:, :, :12, :, :],
        config=[chan.imconfig() for chan in CHANNELS],
    )

    anmi.save("sample.gif", writer="imagemagick", fps=6, extra_args=["-layers", "optimize"])
    plt.close()
    Image("sample.gif")
    ```
    """

    images: list[AxesImage]

    def __init__(
        self,
        data: AnyArray[Batch, Channel, Time, Length, Width] | AnyArray[Channel, Time, Length, Width],
        *,
        config: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # reshape to 5D tensor
        if data.ndim == 4:
            data = data[np.newaxis, ...]
        assert data.ndim == 5

        # construct figure and axes
        B, C, T, L, W = data.shape
        fig, ax = self._subplots(rows=B, channels=C)

        # reshape data to 4D tensor
        data = data.reshape(-1, T, L, W)
        config = config or list(itertools.repeat({}, C))

        images = [
            ax[row, col].imshow(data[row * C + col, 0], **config[col], animated=True)
            for row, col in itertools.product(range(B), range(C))
        ]

        self.data = data
        self.images = images

        super().__init__(fig, self._animate, init_func=lambda: self.images, frames=range(T), blit=True, **kwargs)

    @staticmethod
    def _subplots(rows: int, *, channels: int) -> tuple[Figure, Mapping[tuple[int, int], Axes]]:
        # TODO: determine best scaling factor
        scale = 2

        fig, axes = plt.subplots(rows, channels, figsize=(scale * channels, scale * rows), sharex=True, sharey=True)  # type: ignore
        if rows == 1:
            axes = axes[np.newaxis, ...]
        return fig, axes  # type: ignore

    def _animate(self, tidx: int) -> list[AxesImage]:
        for i in range(len(self.images)):
            self.images[i].set_data(self.data[i, tidx, ...])
        return self.images
