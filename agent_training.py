from __future__ import division

import threading
import time
from math import pi, cos, sin

import level
from draw import Draw, Line


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
    lvl = level.square()
    draw = Draw(lvl.grid.shape)
    threading.Thread(target=simulate, args=(lvl, draw)).start()
    draw.run()


def simulate(lvl, draw):
    for i in range(0, 512):
        lvl.agent.theta -= pi / 64
        draw.update(lvl, mock_lines(lvl.agent))
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
