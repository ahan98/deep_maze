import random
import numpy as np
import pygame
import gymnasium as gym

from gymnasium import spaces
from pygame.surface import Surface
from maze import Maze, ActionSpace
from enum import Enum
from pathfinding import dfs

from mechanics import BaseMazeMechanic, NormalMaze, POMDPMaze

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width=5, height=5, render_mode=None, algorithm="randomized_prim", mode=NormalMaze):

        # maze grid size
        self.size = np.array([width, height])
        # pygame window size (width = 300, proportional height)
        self.window_size = self.size * (300 / height)
        # size of each tile in pixels
        self.tile_size = self.window_size / self.size
        self.maze = Maze(width=width, height=height, maze_algorithm=algorithm)

        self.mode = mode
        self.radius = -1 if mode == NormalMaze else 2

        self.action_space = ActionSpace.space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.uint8),
            "target": spaces.Box(np.zeros(2), self.size - 1, shape=(2,), dtype=np.uint8),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
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

    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Generate new maze
        self.maze.reset()
        # Randomly choose two distinct non-wall locations for agent and target
        agent, target = random.sample(self.maze.open_cells, 2)
        self.agent_location, self.target_location = np.array(agent), np.array(target)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def solve(self):
        n_moves, path = dfs(self.maze, tuple(self.agent_location), tuple(self.target_location))
        print("path =", path)
        ## skip element 0, since that is agent's current position
        for i in range(1, n_moves + 1):
            dx, dy = np.array(path[i]) - self.agent_location
            action = ActionSpace.direction_to_action[(dx, dy)]
            print("action =", action)
            self.step(action)
        print("Shortest path took", n_moves, "moves")

    def step(self, action=None):
        if action is None:
            action = self.action_space.sample()

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = ActionSpace.action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.agent_location = np.clip(
            self.agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.agent_location, self.target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(tuple(self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(tuple(self.window_size))
        # Color.color_canvas(self, canvas)
        # Fill background
        # canvas.fill(Color.GRAY.value)
        # # Color walls
        # for cell in self.maze.wall_cells:
        #     Color.color_tile(canvas, self.tile_size, cell, Color.BLACK)
        # color visible cells
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                cell = (x, y)
                if not self.maze.is_visible(cell, tuple(self.agent_location), self.radius):
                    color = Color.GRAY
                elif np.array_equal(cell, self.target_location):
                    color = Color.GREEN
                elif np.array_equal(cell, self.agent_location):
                    color = Color.RED
                elif self.maze.grid[cell] == 0:
                    color = Color.WHITE
                else:
                    color = Color.BLACK
                Color.color_tile(canvas, self.tile_size, cell, color)

        # canvas.fill(Color.background(self.mode))
        #
        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     Color.GREEN.value,
        #     pygame.Rect(
        #         tuple(self.tile_size * self.target_location),
        #         tuple(self.tile_size)
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.rect(
        #     canvas,
        #     Color.RED.value,
        #     pygame.Rect(
        #         tuple(self.tile_size * self.agent_location),
        #         tuple(self.tile_size)
        #     ),
        # )
        #
        # for x, y in self.maze.wall_cells:
        #     # print("drawing wall at", tuple(self.tile_size * (row, col)))
        #     pygame.draw.rect(
        #         canvas,
        #         Color.BLACK.value,
        #         pygame.Rect(
        #             tuple(self.tile_size * (x, y)),
        #             tuple(self.tile_size)
        #         ),
        #     )
        #
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)

    @staticmethod
    def get_color(env: MazeEnv, cell: tuple[int, int]) -> Enum:
        if not env.maze.is_visible(cell, tuple(env.agent_location), env.radius):
            return Color.GRAY
        elif np.array_equal(cell, env.target_location):
            return Color.GREEN
        elif np.array_equal(cell, env.agent_location):
            return Color.RED
        elif env.maze.grid[cell] == 0:
            return Color.WHITE
        else:
            return Color.BLACK

    @staticmethod
    def color_tile(canvas: Surface, tile_size, cell: tuple[int, int], color: Enum):
        tile = pygame.Rect(tuple(tile_size * cell), tuple(tile_size))
        pygame.draw.rect(canvas, color.value, tile)

    @staticmethod
    def color_canvas(env: MazeEnv, canvas: Surface):
        for (x, y) in env.maze.visible_cells(tuple(env.agent_location), env.radius):
            cell = (x, y)
            color = Color.get_color(env, cell)
            Color.color_tile(canvas, env.tile_size, (x, y), color)
