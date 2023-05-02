import random
from typing import Optional
import numpy as np
from numpy._typing import NDArray
import pygame
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import cv2

from IPython.display import clear_output
from dataclasses import dataclass
from gymnasium import spaces
from enum import Enum

from ..pathfinding import dfs
from ..maze import Maze, ActionSpace


class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)


@dataclass
class Settings:
    radius: int = -1
    time_limit: float = -1
    agent_color: Color = Color.RED
    target_color: Color = Color.GREEN
    fog_color: Color = Color.GRAY
    wall_color: Color = Color.BLACK
    background_color: Color = Color.WHITE
    image_size: tuple[int, int] = (84, 84)  # size in pixels of image for training


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "jupyter"], "render_fps": 4}

    def __init__(self,
                 width: int = 5,
                 height: int = 5,
                 maze_generator: str = "randomized_prim",
                 render_mode: Optional[str] = None,
                 settings: Settings = Settings()):

        # maze grid size
        self.size = np.array([width, height])
        # set pygame window size by rounding height to largest multiple >= 300,
        # and multiply window width by the same factor to maintain aspect ratio
        self.window_size = self.size * np.ceil(300 / height)
        # size of each tile in pixels
        self.tile_size = self.window_size / self.size
        self.maze = Maze(width=width, height=height, maze_algorithm=maze_generator)

        self.action_space = ActionSpace.space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.int64),
            "target": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.int64)
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode != "human":
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.settings = settings

    def _get_obs(self):
        return {"agent": self.agent, "target": self.target}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self.agent - self.target, ord=1),
            "image": cv2.resize(self.rgb_array, dsize=self.settings.image_size)
        }

    def reset(self, seed=None, options=None):
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

        n_moves, path = dfs(self.maze, tuple(self.agent), tuple(self.target))
        print("path =", path)
        ## skip element 0, since that is agent's current position
        for i in range(1, n_moves + 1):
            dx, dy = np.array(path[i]) - self.agent
            action = ActionSpace.direction_to_action[(dx, dy)]
            print("action =", action)
            self.step(action)
        print("Shortest path took", n_moves, "moves")

    def step(self, action):
        """Performs one action for the agent and renders the resulting game state."""

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = ActionSpace.action_to_direction[action]
        nxt = self.agent + direction
        if self.maze.is_legal(*nxt):
            self.agent = nxt

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.agent, self.target)
        reward = 1 if terminated else 0  # Binary sparse rewards

        self.render()
        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        # if self.settings.time_limit > 0:
        #     truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        """Updates canvas based on current game state (maze design and agent position)."""

        if self.render_mode not in self.metadata["render_modes"]:
            return None

        # initialize pygame display for human rendering mode
        if self.render_mode == "human":
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

                if not self.maze.is_visible(cell, tuple(self.agent), self.settings.radius):
                    color = self.settings.fog_color
                elif np.array_equal(cell, self.target):
                    color = self.settings.target_color
                elif np.array_equal(cell, self.agent):
                    color = self.settings.agent_color
                elif self.maze.grid[cell] == 0:
                    color = self.settings.background_color
                else:
                    color = self.settings.wall_color

                tile = pygame.Rect(tuple(self.tile_size * cell), tuple(self.tile_size))
                pygame.draw.rect(canvas, color.value, tile)

        # copy canvas to display
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            self.rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            if self.render_mode == "jupyter":
                clear_output(wait=True)
                plt.imshow(self.rgb_array)
                plt.show()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
