from __future__ import division

import threading
import time

import controller
import level
from draw import Draw, Line
from vision import Vision

_KEY_WALK_FORWARD = 87  # w
_KEY_WALK_BACKWARD = 83  # s
_KEY_TURN_LEFT = 65  # a
_KEY_TURN_RIGHT = 68  # d

_SIGHT_SIGNALS = 32


class Simulator:
    def __init__(self):
        self._lvl = level.square()
        self._draw = Draw(self._lvl.grid.shape, sight_res=_SIGHT_SIGNALS)
        self._ctrl = controller.Controller(self._lvl)
        self._vision = Vision(self._lvl, self._lvl.grid.shape, signal_count=_SIGHT_SIGNALS)

    def run(self, after_init=None, on_key_press=None, on_close=None):

        # prepare first frame
        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))

        if after_init is not None:
            threading.Thread(target=after_init, args=[self, sightline]).start()

        self._draw.run(key_handler=on_key_press, close_handler=on_close)

    @staticmethod
    def _color(channels):
        if channels[0] > 0:
            return [0.8, 0.8, 0.8, channels[0]]
        elif channels[1] > 0:
            return [1.0, 0.843, 0.0, channels[1]]
        else:
            return [0.0, 0.0, 0.0, 0.0]

    def _lines(self, signals):
        signal_lines = []
        for signal in signals:
            signal_lines.append(Line(signal.a, signal.b, self._color(signal.channels)))
        return signal_lines

    def _sight_colors(self, signals):
        colors = []
        for signal in signals:
            colors.append(self._color(signal.channels))
        return colors

    def step(self, action):
        self._ctrl.step(action)
        sightline = self._vision.look()
        self._draw.update(self._lvl, self._lines(sightline), self._sight_colors(sightline))
        return sightline


if __name__ == "__main__":

    action = [None]
    done = [False]


    def train(sim, first_input):
        # track fps
        timer = time.time()
        frames = 0

        while not (done[0]):

            # print fps
            frames += 1
            time_delta = time.time() - timer
            fps_interval = 10.0
            if time_delta >= fps_interval:
                timer += fps_interval
                print 'fps over %ss: %s' % (fps_interval, frames / time_delta)
                frames = 0

            # TODO action should be read & updated atomically
            act = action[0]
            if act is not None:
                sim.step(act)
                action[0] = None
            else:
                time.sleep(1 / 60)


    def key_press(symbol, modifiers):
        if symbol == _KEY_WALK_FORWARD:
            action[0] = controller.walk_forward
        elif symbol == _KEY_WALK_BACKWARD:
            action[0] = controller.walk_backward
        elif symbol == _KEY_TURN_LEFT:
            action[0] = controller.turn_left
        elif symbol == _KEY_TURN_RIGHT:
            action[0] = controller.turn_right


    def close():
        done[0] = True


    sim = Simulator()
    sim.run(train, key_press, close)
