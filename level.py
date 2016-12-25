from math import pi
import numpy as np


def square():
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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

    agent = Agent(2.0, 2.0, pi / 4.0)

    return Level(grid, coins, agent)


class Agent:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta


class Coin:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Level:
    agent = None
    coins = None

    def __init__(self, grid=np.empty((0, 0)), coins=[], agent=Agent()):
        self.grid = grid
        self.coins_origin = coins
        self.agent_origin = agent
        self.reset()

    def reset(self):
        self.coins = self.coins_origin
        self.agent = self.agent_origin
