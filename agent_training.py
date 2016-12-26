from __future__ import division

import random
import threading
import time
from math import pi, cos, sin

import controller
import level
from draw import Draw, Line


_KEY_WALK_FORWARD = 87  # w
_KEY_TURN_LEFT = 65     # a
_KEY_TURN_RIGHT = 68    # d

_action = [None]
_done = [False]

def mock_lines(agent):
    length = 8
    total = 32
    arc_length = pi / 2
    theta_min = agent.theta - (arc_length / 2)
    theta_step = arc_length / (total - 1)
    agent_radius = 0.45
    lines = []
    theta = theta_min
    for i in range(0, total):
        a = [agent_radius * cos(theta) + agent.coord[0], agent_radius * sin(theta) + agent.coord[1]]
        b = [length * cos(theta) + agent.coord[0], length * sin(theta) + agent.coord[1]]
        color = [0.0, 0.0, 0.0]
        lines.append(Line(a, b, color))
        theta += theta_step
    return lines


def main():
    lvl = level.collisions()
    draw = Draw(lvl.grid.shape, level_scale=1)
    ctrl = controller.Controller(lvl)
    threading.Thread(target=simulate, args=(lvl, ctrl, draw)).start()
    draw.run(key_handler=on_key_press, close_handler=on_close)


def simulate(lvl, ctrl, draw):
    draw.update(lvl, mock_lines(lvl.agent))
    while not(_done[0]):
        if _action[0] is not None:
            ctrl.step(_action[0])
            _action[0] = None
        draw.update(lvl, mock_lines(lvl.agent))
        time.sleep(1 / 60)


def on_key_press(symbol, modifiers):
    if symbol == _KEY_WALK_FORWARD:
        _action[0] = controller.walk_forward
    elif symbol == _KEY_TURN_LEFT:
        _action[0] = controller.turn_left
    elif symbol == _KEY_TURN_RIGHT:
        _action[0] = controller.turn_right


def on_close():
    _done[0] = True

if __name__ == "__main__":
    main()
