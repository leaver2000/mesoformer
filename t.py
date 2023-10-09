import types
import pandas as pd
from typing import Any

# pd.Series.__class_getitem__ = classmethod(types.GenericAlias)
# x: pd.Series[int]
# z: classmethod[Any, Any, Any]
# # from __future__ import annotations

# # from src.mesoformer.enums import Dimensions


# # Dimensions.remap(["x"])
def set_generic_alias(t: type):
    if not hasattr(t, "__class_getitem__"):
        setattr(t, "__class_getitem__", classmethod(types.GenericAlias))


try:
    s1: pd.Series[int] = pd.Series([1, 2, 3])  # raises TypeError
    raise RuntimeError("Should have failed")
except TypeError:
    ...
set_generic_alias(pd.Series)

# set_generic_alias(classmethod)
# c2: classmethod[Any, Any, Any]  # Ok
