import numpy as np

from queue import PriorityQueue

from .types import *
from .maze import Maze


def dfs(maze: Maze, start: Coord, goal: Coord) -> list[Coord]:
    """
    depth-first-search
    :param maze_game: the GameMaze instance
    :param start: tuple (x,y) of start position
    :param goal: tuple (x,y) of the goal position
    :return: list containing the path
    """
    visited = np.full(maze.grid.shape, False)
    visited[start[0], start[1]] = True

    shortest_path = []

    pq = PriorityQueue()
    pq.put((1, [start]))
    while pq:
        length, path = pq.get()
        x, y = path[-1]
        if x == goal[0] and y == goal[1]:
            shortest_path = path
            break

        visited[x, y] = True

        # get legal unvisited neighbors
        for nx, ny in maze.legal_cells(x, y):
            if visited[nx, ny]:
                continue
            pq.put((length + 1, path + [(nx, ny)]))

    return shortest_path
