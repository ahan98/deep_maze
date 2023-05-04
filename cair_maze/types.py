from __future__ import annotations
from typing import Annotated, Optional, Union
from nptyping import NDArray, Shape, Int, UInt, Float32, Float64
from annotated_types import Ge

# Real = Union[Int, UInt, Float32, Float64]
# Natural = Annotated[int, Ge(0)]
Grid = NDArray[Shape["*, *"], UInt]
Coord = tuple[int, int]|NDArray[Shape["2"], Int]
