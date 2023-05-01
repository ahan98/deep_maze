# import gymnasium as gym
import gym
from gym.envs.registration import register
import gym_maze.envs.maze_env

cls_mg = gym_maze.envs.maze_env.MazeGame
cls_base = gym_maze.envs.maze_env.MazeEnv

sizes = [(a, a) for a in range(5, 56)]
# print(sizes[:5])
mechanics = [
    [cls_mg.NormalMaze, [dict()]],
    [cls_mg.POMDPMaze, [dict(vision=3, show_target=False)]],
    [cls_mg.POMDPLimitedMaze, [dict(vision=3, show_target=False)]],
    [cls_mg.TimedPOMDPMaze, [dict(vision=3, show_target=False, delay=5)]],
    [cls_mg.TimedPOMDPLimitedMaze, [dict(vision=3, show_target=False, delay=5)]]
]


def make_constructor(args):
    def constructor(self):
        super(self.__class__, self).__init__(*args)

    return constructor


for size in sizes:
    for mechanic, mechanic_args in mechanics:
        for mechanic_arg in mechanic_args:
            data = (size[0], size[1], str(mechanic.__name__))
            cls_name = "Maze%sx%s%sEnv" % data
            cls_id = "Maze-%sx%s-%s-v0" % data  # maze-11x11-deterministic-v0
            cls_constructor = make_constructor(args=(*size, mechanic, mechanic_arg))

            cls = type(cls_name, (cls_base,), {
                "__init__": cls_constructor
            })
            cls.id = cls_id
            # print("cls_id", cls_id)
            register(
                id=cls.id,
                entry_point='gym_maze.envs:%s' % cls_name
            )

            setattr(gym_maze.envs, cls_name, cls)

# # NoMaze Environment
#
# sizes = [(a, a) for a in range(2, 56, 2)]
# cls_base = gym_maze.envs.NoMazeEnv
# for size in sizes:
#     # Register NoMaze Environments
#     data = (size[0], size[1])
#     cls_name = "NoMaze%sx%sEnv" % data
#     cls_id = "NoMaze-%sx%s-v0" % data  # maze-11x11-deterministic-v0
#     cls_constructor = make_constructor(args=(*size, ))
#     cls = type(cls_name, (cls_base,), {
#         "__init__": cls_constructor
#     })
#     cls.id = cls_id
#     register(
#         id=cls.id,
#         entry_point='gym_maze.envs:%s' % cls_name
#     )
#
#     setattr(gym_maze.envs, cls_name, cls)
