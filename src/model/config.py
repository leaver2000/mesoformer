import dataclasses
from mesoscaler._typing import Pair
from typing import NamedTuple


class Shape(NamedTuple):
    channels: int
    t: int
    z: int
    y: int
    x: int


@dataclasses.dataclass(
    kw_only=True,
)
class Config:
    dimensions: Shape
    patch: Shape
    # width: int
    # height: int
    # dx: int
    # dy: int
    # ratio: float

    # time_n: int = 1
    # time_patch_size: int = 1

    # @classmethod
    # def from_ratio(cls, img_size: int, distance: int, ratio: float):
    #     height = int(img_size / ratio)
    #     width = img_size
    #     dx = distance
    #     dy = int(dx / ratio)
    #     return cls(
    #         width=width,
    #         height=height,
    #         dx=dx,
    #         dy=dy,
    #         ratio=ratio,
    #     )

    # @property
    # def image_size(self) -> Pair[int]:
    #     return (self.width, self.height)

    # @property
    # def patch_size(self) -> Pair[int]:
    #     return (self.dx, self.dy)

    def _validate(self):
        for x, y in zip(self.dimensions, self.patch):
            assert x % y == 0

    #     img_size = self.image_size
    #     patch_size = self.patch_size
    #     frames = self.time_n
    #     t_patch_size = self.time_n
    #     assert img_size[1] % patch_size[1] == 0
    #     assert img_size[0] % patch_size[0] == 0
    #     assert frames % t_patch_size == 0


Config(
    dimensions=Shape(5, 4, 10, 40, 80),
    patch=Shape(5, 1, 10, 40, 40),
)
