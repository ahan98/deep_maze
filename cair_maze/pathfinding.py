import numpy as np
from queue import PriorityQueue
from .types import *
from .maze import Maze


def dfs(maze: Maze, start: Coord, goal: Coord):
    """
    depth-first-search
    :param maze_game: the GameMaze instance
    :param start: tuple (x,y) of start position
    :param goal: tuple (x,y) of the goal position
    :return: list containing the path
    """
    start = tuple(start)
    stack = [(start, [start])]
    possible_path = PriorityQueue()
    visited = np.zeros_like(maze.grid)

    while stack:
        (vertex, path) = stack.pop()
        visited[vertex] = 1

        # get legal unvisited neighbors
        for next in maze.legal_cells(*vertex):
            if visited[next]:
                continue
            if np.array_equal(next, goal):
                full_path = path + [next]
                length = len(path)
                possible_path.put((length, full_path))
            else:
                stack.append((next, path + [next]))

    return possible_path.get()
