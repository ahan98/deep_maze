from __future__ import annotations
from typing import Annotated, Optional, Union
from nptyping import NDArray, Shape, Int, UInt, Float32, Float64
from annotated_types import Ge

# Real = Union[Int, UInt, Float32, Float64]
Grid = NDArray[Shape["*, *"], UInt]
Natural = Annotated[int, Ge(0)]
Coord = tuple[int, int]|NDArray[Shape["2"], Int]

# class Coord(tuple[int, int]):
#
#     def __new__(cls, x: int, y: int):
#         return super(Coord, cls).__new__(cls, (x, y))
#
#     def __init__(self, x: int, y: Optional[int] = None):
#         if y is None:
#             y = x
#         self.x = x
#         self.y = y
#
#     def __sub__(self, other: CoordLike|int):
#         if type(other) == int:
#             x = self.x - other
#             y = self.x - other
#         else:
#             x = self.x - other[0]
#             y = self.y - other[1]
#
#         return Coord(x, y)
#
#     def __add__(self, other: CoordLike|int):
#         if type(other) == int:
#             x = self.x + other
#             y = self.x + other
#         else:
#             x = self.x + other[0]
#             y = self.y + other[1]
#
#         return Coord(x, y)
#
#     def __mul__(self, other: CoordLike|int):
#         if type(other) == int:
#             x = self.x * other
#             y = self.y * other
#         else:
#             x = self.x * other[0]
#             y = self.y * other[1]
#         return Coord(x, y)

# CoordLike = Coord|tuple[int, int]|NDArray[Shape["2"], Int]
