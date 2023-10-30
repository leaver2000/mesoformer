# reduce.py
from __future__ import annotations

import itertools
import operator

map_reduce = lambda f: lambda x, y: itertools.starmap(f, zip(x, y))
truediv = map_reduce(operator.truediv)
floordiv = map_reduce(operator.floordiv)
equals = map_reduce(operator.eq)
mod = map_reduce(operator.mod)
pow_ = map_reduce(operator.pow)
add = map_reduce(operator.add)
sub = map_reduce(operator.sub)
