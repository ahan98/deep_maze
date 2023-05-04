import time
import numpy as np

from typing import Optional

from .algorithms import backtracking, randomized_prim
from .spaces import ActionSpace
from .types import *


class Maze:
    """
    Maze Class, Creates a Maze Instance that contains the internal data of the maze.
    """
    def __init__(self, width: p_int=15, height: p_int=15, seed_action=time.time(),
                 algorithm: Optional[str]="randomized_prim"):
        """
        Maze Instance, Contains maze generator and the data related to it
        :param width: width of the maze in tiles
        :param height: height of the maze in tiles
        :param seed_action: seed of the action sampler
        :param algorithm: the generator algorithm. currently supported: randomized_prim
        """

        self.width = width
        self.height = height
        self.grid: Grid = np.zeros((width, height), dtype=np.uint)
        self.algorithm = algorithm
        self.reset()

    def legal_cells(self, x: u_int, y: u_int) -> list[Coord]:
        cells = []
        for (dx, dy) in ActionSpace.directions:
            nxt = (x + dx, y + dy)
            if self.is_legal(*nxt):
                cells.append(nxt)
        return cells

    def is_legal(self, x: u_int, y: u_int) -> bool:
        return self.inbounds(x, y) and self.grid[x, y] == 0

    def inbounds(self, x: u_int, y: u_int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_visible(self, cell: Coord, center: Coord, radius: Optional[p_int]=None) -> bool:
        if radius is None:
            return True

        return center[0] - radius <= cell[0] <= center[0] + radius and \
               center[1] - radius <= cell[1] <= center[1] + radius and \
               self.inbounds(*cell)

    def visible_cells(self, center: Coord, radius: Optional[p_int]=None) -> list[Coord]:
        if radius is None:
            return self.wall_cells + self.open_cells

        cells = []
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if self.inbounds(x, y):
                    cells.append((x, y))
        return cells

    def reset(self):
        # Generate the maze structure
        self._generate()
        self.wall_cells: list[Coord] = list(zip(*np.where(self.grid == 1)))
        self.open_cells: list[Coord] = list(zip(*np.where(self.grid == 0)))

    def _generate(self):
        """
        Generates the maze based on which algorithm was defined in the constructor
        :return: None
        """
        match self.algorithm:
            case "backtracking":
                backtracking(self.grid)
            case "randomized_prim":
                randomized_prim(self.grid)
            case None:
                pass
            case _:
                raise Exception("Undefined maze generation algorithm:", self.algorithm)
