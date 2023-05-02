import time
import numpy as np

from typing import Optional
from gymnasium import spaces

from .algorithms import recursive_backtracking, randomized_prim


class ActionSpace:
    space = spaces.Discrete(4)
    # up, down, left, right
    directions: list[tuple] = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    actions: list[int] = [0, 1, 2, 3]
    direction_to_action = dict(zip(directions, actions))
    action_to_direction = dict(zip(actions, directions))


class Maze:
    """
    Maze Class, Creates a Maze Instance that contains the internal data of the maze.
    """
    def __init__(self, width=15, height=15, seed_action=time.time(), maze_algorithm="randomized_prim"):
        """
        Maze Instance, Contains maze generator and the data related to it
        :param width: width of the maze in tiles
        :param height: height of the maze in tiles
        :param seed_action: seed of the action sampler
        :param maze_algorithm: the generator algorithm. currently supported: randomized_prim
        """

        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        # self.action_space = ActionSpace(seed=seed_action)
        # self.state_space = StateSpace(self)
        self.maze_algorithm = maze_algorithm
        self.reset()

    def legal_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        cells = []
        for (dx, dy) in ActionSpace.directions:
            nxt = (x + dx, y + dy)
            if self.is_legal(*nxt):
                cells.append(nxt)
        return cells

    def is_legal(self, x: int, y: int) -> bool:
        return self.inbounds(x, y) and self.grid[x, y] == 0

    def inbounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_visible(self, cell: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
        if radius == -1:
            return True

        return center[0] - radius <= cell[0] <= center[0] + radius and \
               center[1] - radius <= cell[1] <= center[1] + radius and \
               self.inbounds(*cell)

    def visible_cells(self, center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        if radius == -1:
            return [(x, y) for x in range(self.width) for y in range(self.height)]

        cells = []
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if self.inbounds(x, y):
                    cells.append((x, y))
        return cells


    def reset(self):
        # Generate the maze structure
        self._generate()
        self.wall_cells: list[tuple[int, int]] = list(zip(*np.where(self.grid == 1)))
        self.open_cells: list[tuple[int, int]] = list(zip(*np.where(self.grid == 0)))

    def _generate(self):
        """
        Generates the maze based on which algorithm was defined in the constructor
        :return: None
        """
        match self.maze_algorithm:
            case "recursive_backtracking":
                recursive_backtracking(self.grid)
            case "randomized_prim":
                randomized_prim(self.grid)
            case None:
                pass
            case _:
                raise Exception("Undefined maze generation algorithm:", self.maze_algorithm)
