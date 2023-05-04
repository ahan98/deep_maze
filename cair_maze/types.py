from __future__ import annotations
from typing import Annotated, Optional, Union
from nptyping import NDArray, Shape, UInt, Int, Float32, Float64
from annotated_types import Ge, Gt

u_int = Annotated[int, Ge(0)]
p_int = Annotated[int, Gt(0)]
Grid = NDArray[Shape["*, *"], UInt]
Coord = tuple[u_int, u_int]|NDArray[Shape["2"], UInt]
