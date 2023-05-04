from gymnasium import spaces


class ActionSpace:
    space = spaces.Discrete(4)
    # up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    actions = [0, 1, 2, 3]
    direction_to_action = dict(zip(directions, actions))
    action_to_direction = dict(zip(actions, directions))
