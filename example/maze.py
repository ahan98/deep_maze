# import gymnasium as gym
import gym
import random

import time
import pygame

import gym_maze
import cair_maze
from cair_maze.maze_game import StateType
import pygame

# from cair_maze.maze_game import MazeGame

if __name__ == "__main__":
    pygame.display.set_mode((640, 480))
    # env = gym.make("gym_maze:Maze-32x32-NormalMaze-v0")
    env = gym.make("gym_maze:Maze-32x32-POMDPMaze")
    env.reset()
    terminal = False
    while not terminal:
        a = env.action_space.sample()
        observation, reward, terminal, info = env.step(a)
        print(observation, reward, terminal, info)
        env.render()
        if terminal:
            env.reset()
    env.close()

    # # Direct initialization
    # m = MazeGame((32, 32), mechanic=MazeGame.NormalMaze, mechanic_args=dict(vision=3))
    # path = list(reversed(m.maze_optimal_path[1]))
    #
    # fps = 0
    # path.pop()
    #
    # while True:
    #     nx, ny = path.pop()
    #     px, py = m.player
    #     dx, dy = nx - px, ny - py
    #
    #     if dx == 1:
    #         a = 3
    #     elif dx == -1:
    #         a = 2
    #     elif dy == 1:
    #         a = 0
    #     elif dy == -1:
    #         a = 1
    #     else:
    #         Exception("omg")
    #
    #     m.render()
    #     data = m.step(a)
    #     m.render()
    #     if m.terminal:
    #         m.reset()
    #         path = list(reversed(m.maze_optimal_path[1]))
    #
    #         path.pop()

