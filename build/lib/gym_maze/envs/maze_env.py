# import gymnasium as gym
import gym
from gym import spaces
from cair_maze.maze_game import MazeGame, StateType
import numpy as np
import pygame

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    id = "maze-v0"

    def __init__(self, width, height, screen_size, mechanic, mechanic_args, render_mode=None):
        self.env = MazeGame((width, height), screen_size=screen_size, mechanic=mechanic, mechanic_args=mechanic_args)
        self.observation_space = spaces.Dict({
            "player": spaces.Box(np.array([0, 0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int),
            "target": spaces.Box(np.array([0, 0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int),
        })
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.window_size = 512
        self.window = None
        self.clock = None

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode(
    #             (self.window_size, self.window_size)
    #         )
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()


    def get_state(self):
        return self.env.get_state()

    def step(self, action, type=StateType.DEFAULT):
        return self.env.step(action, type)

    def reset(self):
        return self.env.reset()

    def render(self, mode=0, close=False):
        if close:
            self.env.quit()
            return None

        return self.env.render(type=mode)


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

