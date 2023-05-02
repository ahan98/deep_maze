from queue import PriorityQueue
from maze import Maze


def dfs(maze: Maze, start, goal):
    """
    depth-first-search
    :param maze_game: the GameMaze instance
    :param start: tuple (x,y) of start position
    :param goal: tuple (x,y) of the goal position
    :return: list containing the path
    """
    stack = [(start, [start])]
    possible_path = PriorityQueue()

    while stack:
        (vertex, path) = stack.pop()
        # get legal unvisited neighbors
        legal_cells = set(maze.legal_cells(*vertex)) - set(path)
        for next in legal_cells:
            if next == goal:
                full_path = path + [next]
                length = len(path)
                possible_path.put((length, full_path))
            else:
                stack.append((next, path + [next]))

    return possible_path.get()

