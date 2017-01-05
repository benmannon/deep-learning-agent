from math import pi

import numpy as np


def square():
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    coins = [
        [2.0, 7.0],
        [2.0, 8.0],
        [3.0, 8.0],
        [4.0, 4.0],
        [4.0, 6.0],
        [6.0, 4.0],
        [6.0, 6.0],
        [7.0, 2.0],
        [7.0, 8.0],
        [8.0, 2.0],
        [8.0, 3.0],
        [8.0, 7.0],
        [8.0, 8.0]
    ]

    agent = Agent([2.0, 2.0], pi / 4.0)

    return Level(grid, coins, agent, 200)


def collisions():
    grid = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1]
    ])

    coins = []

    agent = Agent([2.0, 2.0], pi / 4.0)

    return Level(grid, coins, agent, 100)


class Agent:

    coord = None
    theta = None

    def __init__(self, coord=[0.0, 0.0], theta=0.0):
        self._coord_origin = np.copy(coord)
        self._theta_origin = theta
        self.reset()

    def reset(self):
        self.coord = np.copy(self._coord_origin)
        self.theta = np.copy(self._theta_origin)


class Level:
    agent = None
    coins = None

    def __init__(self, grid, coins, agent, time):
        self.grid = grid
        self._coins_origin = np.copy(coins)
        self.agent = agent
        self.time = time
        self.reset()

    def reset(self):
        self.coins = np.copy(self._coins_origin).tolist()
        self.agent.reset()
