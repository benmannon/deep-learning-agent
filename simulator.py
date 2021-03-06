from __future__ import division

import threading
from collections import namedtuple

import level
import vision
from controller import Controller
from draw import Draw, Line
from vision import Vision

CHANNEL_NUM = vision.CHANNEL_NUM


def _color(channels, args):
    if channels[0] > 0:
        return args.grid_color + [channels[0]]
    elif channels[1] > 0:
        return args.coin_color + [channels[1]]
    else:
        return [0.0, 0.0, 0.0, 0.0]


def _lines(signals, args):
    signal_lines = []
    for signal in signals:
        ax, ay = signal.ax, signal.ay
        bx, by = signal.bx, signal.by
        r, g, b, a = _color(signal.channels, args)
        signal_lines.append(Line(ax, ay, bx, by, r, g, b, a))
    return signal_lines


def _sight_colors(signals, args):
    colors = []
    for signal in signals:
        colors.append(_color(signal.channels, args))
    return colors


def _channels(signals):
    channels = []
    for signal in signals:
        channels.append(signal.channels)
    return channels


class Simulator:
    def __init__(self, args):

        self._args = args
        self._lvl = level.square()

        self._ctrl = Controller(args, self._lvl)
        self._vision = Vision(args, self._lvl)
        self._sightline = self._vision.look()

        self._draw = Draw(args, self._lvl.grid.shape, self._draw_update)

        self._time_step = self._lvl.time

    def run(self, after_init=None, on_key_press=None, on_close=None):

        if after_init is not None:
            sight_channels = _channels(self._sightline)
            state = _State(-1, sight_channels)
            threading.Thread(target=after_init, args=[self, state]).start()

        self._draw.run(key_handler=on_key_press, close_handler=on_close)

    def step(self, action):

        self._time_step -= 1

        # take action and check for rewards
        coins_available = len(self._lvl.coins)
        is_colliding = self._ctrl.step(action)
        coins_left = len(self._lvl.coins)
        coins_collected = coins_available - coins_left
        reward = float(coins_collected) * self._args.reward_coin
        end = False

        if is_colliding:
            reward += self._args.reward_collision

        # no more coins? out of time? reset the level
        if coins_left == 0:
            reward += self._args.reward_win
            self._time_step = self._lvl.time
            self._lvl.reset()
            end = True
        elif self._time_step <= 0:
            reward += self._args.reward_loss
            self._time_step = self._lvl.time
            self._lvl.reset()
            end = True

        self._sightline = self._vision.look()
        state = _State(action, _channels(self._sightline))
        return state, reward, end

    def _draw_update(self):
        grid = self._lvl.grid
        agent_coord = list(self._lvl.agent.coord)
        agent_theta = self._lvl.agent.theta
        coins = list(self._lvl.coins)
        lines = _lines(self._sightline, self._args)
        sight_colors = _sight_colors(self._sightline, self._args)
        return grid, agent_coord, agent_theta, coins, lines, sight_colors

_State = namedtuple('State', 'prev_action vision')
