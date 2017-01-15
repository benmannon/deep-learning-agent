from __future__ import division

import threading

import controller
import level
import vision
from draw import Draw, Line
from vision import Vision

CHANNEL_NUM = vision.CHANNEL_NUM


class Simulator:
    def __init__(self, args):

        self._reward_coin = args.reward_coin
        self._reward_win = args.reward_win
        self._reward_loss = args.reward_loss
        self._reward_collision = args.reward_collision

        self._lvl = level.square()
        self._grid_color = args.grid_color
        self._coin_color = args.coin_color

        self._draw = Draw(args, self._lvl.grid.shape)
        self._ctrl = controller.Controller(args, self._lvl)

        self._vision = Vision(self._lvl,
                              agent_radius=args.agent_radius,
                              coin_radius=args.coin_radius,
                              signal_count=args.agent_vision_res,
                              fov=args.agent_vision_fov,
                              attenuation=args.agent_vision_attenuation)

        self._time_step = self._lvl.time

    def run(self, after_init=None, on_key_press=None, on_close=None):

        # prepare first frame
        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))

        if after_init is not None:
            sight_channels = self._channels(sightline)
            threading.Thread(target=after_init, args=[self, sight_channels]).start()

        self._draw.run(key_handler=on_key_press, close_handler=on_close)

    def _color(self, channels):
        if channels[0] > 0:
            return self._grid_color + [channels[0]]
        elif channels[1] > 0:
            return self._coin_color + [channels[1]]
        else:
            return [0.0, 0.0, 0.0, 0.0]

    def _lines(self, signals):
        signal_lines = []
        for signal in signals:
            a = [signal.ax, signal.ay]
            b = [signal.bx, signal.by]
            signal_lines.append(Line(a, b, self._color(signal.channels)))
        return signal_lines

    def _sight_colors(self, signals):
        colors = []
        for signal in signals:
            colors.append(self._color(signal.channels))
        return colors

    def step(self, action):

        self._time_step -= 1

        # take action and check for rewards
        coins_available = len(self._lvl.coins)
        is_colliding = self._ctrl.step(action)
        coins_left = len(self._lvl.coins)
        coins_collected = coins_available - coins_left
        reward = float(coins_collected) * self._reward_coin
        end = False

        if is_colliding:
            reward += self._reward_collision

        # no more coins? out of time? reset the level
        if coins_left == 0:
            reward += self._reward_win
            self._time_step = self._lvl.time
            self._lvl.reset()
            end = True
        elif self._time_step <= 0:
            reward += self._reward_loss
            self._time_step = self._lvl.time
            self._lvl.reset()
            end = True

        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))
        return self._channels(sightline), reward, end

    @staticmethod
    def _channels(signals):
        channels = []
        for signal in signals:
            channels.append(signal.channels)
        return channels
