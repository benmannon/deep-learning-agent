from math import pi, cos, sin

walk_forward = 0
turn_left = 1
turn_right = 2


class Controller:

    _stride = 0.1
    _turn = pi / 8

    def __init__(self, level):
        self._level = level

    def step(self, action):
        agent = self._level.agent
        if action == walk_forward:
            x, y = agent.coord[0], agent.coord[1]
            theta = agent.theta
            stride = self._stride
            agent.coord = [x + stride * cos(theta), y + stride * sin(theta)]
        elif action == turn_left:
            agent.theta += self._turn
        elif action == turn_right:
            agent.theta -= self._turn
