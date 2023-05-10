import random
from typing import Optional
from .types import *
from .spaces import ActionSpace


def randomized_prim(grid: Grid, start: Coord=(0, 0), seed=None) -> None:
    width, height = grid.shape

    # Start with a grid filled with walls
    grid.fill(1)

    frontiers = [(start, start)]

    while frontiers:

        random.Random(seed).shuffle(frontiers)
        cell = frontiers.pop()
        x, y = cell[1]

        if grid[x, y] == 1:
            grid[cell[0]] = 0
            grid[x, y] = 0

            if x >= 2 and grid[x-2, y] == 1:
                frontiers.append(((x-1, y), (x-2, y)))

            if x < width-2 and grid[x+2, y] == 1:
                frontiers.append(((x+1, y), (x+2, y)))

            if y >= 2 and grid[x, y-2] == 1:
                frontiers.append(((x, y-1), (x, y-2)))

            if y < height-2 and grid[x, y+2] == 1:
                frontiers.append(((x, y+1), (x, y+2)))


def backtracking(grid: Grid, start: Optional[Coord]=None, seed=None):
    """ https://en.wikipedia.org/wiki/Maze_generation_algorithm """
    width, height = grid.shape
    directions = ActionSpace.directions

    if start is None:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
    else:
        x, y = start

    grid.fill(1)
    cells = [(x, y)]
    while cells:
        x, y = cells.pop()
        grid[x, y] = 0
        random.Random(seed).shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + 2*dx, y + 2*dy
            if 0 <= nx < width and 0 <= ny < height and grid[nx, ny] == 1:
                grid[x + dx, y + dy] = 0
                cells.append((x, y))
                cells.append((nx, ny))
                break
