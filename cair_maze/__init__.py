from gymnasium.envs.registration import register

register(
    id="MazeEnv-v0",
    entry_point="cair_maze.envs:MazeEnv",
)

