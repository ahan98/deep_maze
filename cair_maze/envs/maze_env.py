import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
import gymnasium as gym

from IPython.display import clear_output
from dataclasses import dataclass
from gymnasium import spaces
from enum import Enum
from typing import Optional

from ..types import *
from ..pathfinding import dfs
from ..maze import Maze
from ..spaces import ActionSpace


class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)


@dataclass
class Colors:
    agent: Color = Color.RED
    target: Color = Color.GREEN
    fog: Color = Color.GRAY
    wall: Color = Color.BLACK
    background: Color = Color.WHITE


@dataclass
class MazeOptions:
    width: int = 5
    height: int = 5
    algorithm: Optional[str] = "randomized_prim"
    render_mode: Optional[str] = None
    colors: Colors = Colors()
    image_size: tuple[u_int, u_int] = (84, 84)
    window_height: u_int = 300
    radius: Optional[p_int] = None


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["window", "jupyter"], "render_fps": 4}

    def __init__(self, options: MazeOptions=MazeOptions()):
        # maze grid size
        self.size = np.array([options.width, options.height])
        # set pygame window size by rounding height to largest multiple >= 300,
        # and multiply window width by the same factor to maintain aspect ratio
        aspect_ratio = int(np.ceil(options.window_height / options.width))
        self.window_size = self.size * aspect_ratio
        # size of each square tile in pixels
        self.tile_size = np.ones(2) * aspect_ratio
        self.maze = Maze(width=options.width, height=options.height, algorithm=options.algorithm)

        self.action_space = ActionSpace.space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.int64),
            "target": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.int64)
        })

        assert options.render_mode is None or options.render_mode in self.metadata["render_modes"]
        if options.render_mode != "window":
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.render_mode = options.render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.colors = options.colors
        self.radius = options.radius
        self.image_size = options.image_size

    def _get_obs(self):
        return {"agent": self.agent, "target": self.target}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self.agent - self.target, ord=1),
            "image": cv2.resize(self.rgb_array, dsize=self.image_size)
        }

    def reset(self, seed=None, options: Optional[MazeOptions]=None):
        """Reset the game state to a random maze and agent/target positions."""

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Generate new maze
        self.maze.reset()
        # Randomly choose two distinct non-wall locations for agent and target
        agent, target = random.sample(self.maze.open_cells, 2)
        self.agent, self.target = np.array(agent), np.array(target)

        self.render()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def solve(self):
        """Return shortest path from agent's current position to target using DFS."""

        n_moves, path = dfs(self.maze, self.agent, self.target)
        # skip element 0, since that is agent's current position
        for i in range(1, n_moves + 1):
            direction = tuple(path[i] - self.agent)
            action = ActionSpace.direction_to_action[direction]
            print("action =", action)
            self.step(action)
        print("Shortest path took", n_moves, "moves")

    def step(self, action):
        """Performs one action for the agent and renders the resulting game state."""

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = ActionSpace.action_to_direction[action]
        next_cell = self.agent + direction
        if self.maze.is_legal(*next_cell):
            self.agent = next_cell

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.agent, self.target)
        reward = int(terminated)  # Binary sparse rewards

        self.render()
        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        # if self.settings.time_limit > 0:
        #     truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        """ Draw canvas for current game state. """

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        # initialize pygame display for human rendering mode
        if self.render_mode == "window":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(tuple(self.window_size))

            if self.clock is None:
                self.clock = pygame.time.Clock()

        # draw new canvas
        canvas = pygame.Surface(tuple(self.window_size))
        for x in range(self.maze.width):
            for y in range(self.maze.height):
                cell = (x, y)

                if not self.maze.is_visible(cell, self.agent, self.radius):
                    color = self.colors.fog
                elif np.array_equal(cell, self.target):
                    color = self.colors.target
                elif np.array_equal(cell, self.agent):
                    color = self.colors.agent
                elif self.maze.grid[x, y] == 0:
                    color = self.colors.background
                else:
                    color = self.colors.wall

                tile = pygame.Rect(tuple(self.tile_size * cell), tuple(self.tile_size))
                pygame.draw.rect(canvas, color.value, tile)

        self.rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        # copy canvas to display
        if self.render_mode == "window":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "jupyter":
                clear_output(wait=True)
                plt.imshow(self.rgb_array)
                plt.show()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
