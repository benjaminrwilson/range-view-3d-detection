"""Mathematical constants."""

import sys
from math import pi, tau
from typing import Final

# Difference between 1.0 and the least value greater than 1.0
# that is representable as a float.
# https://docs.python.org/3/library/sys.html#sys.float_info
EPS: Final[float] = sys.float_info.epsilon

INF: Final[float] = float("inf")  # Infinity.
NAN: Final[float] = float("nan")  # Not a number.
PI: Final[float] = pi  # 3.14159 ...
TAU: Final[float] = tau
