import numpy as np

from collections import deque

from .types import *
from .maze import Maze


def bfs(maze: Maze, start: Coord, goal: Coord) -> list[Coord]:
    """
    breadth-first-search
    :param maze_game: the GameMaze instance
    :param start: tuple (x,y) of start position
    :param goal: tuple (x,y) of the goal position
    :return: list containing the path
    """
    visited = np.full(maze.grid.shape, False)
    visited[start[0], start[1]] = True

    shortest_path = []

    q = deque()
    q.append([start])
    while q:
        path = q.popleft()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            shortest_path = path
            break

        visited[x, y] = True

        # get legal unvisited neighbors
        for nx, ny in maze.legal_cells(x, y):
            if visited[nx, ny]:
                continue
            q.append(path + [(nx, ny)])

    return shortest_path
