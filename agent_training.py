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

_action = [None]
_done = [False]


def main():
    lvl = level.square()
    draw = Draw(lvl.grid.shape, level_scale=1)
    ctrl = controller.Controller(lvl)
    vision = Vision(lvl, lvl.grid.shape)
    threading.Thread(target=simulate, args=(lvl, ctrl, vision, draw)).start()

    # TODO fix this race condition
    time.sleep(2)

    draw.run(key_handler=on_key_press, close_handler=on_close)


def color(channels):
    if channels[0] > 0:
        return [0.8, 0.8, 0.8, channels[0]]
    elif channels[1] > 0:
        return [1.0, 0.843, 0.0, channels[1]]
    else:
        return [0.0, 0.0, 0.0, 0.0]


def lines(signals):
    signal_lines = []
    for signal in signals:
        signal_lines.append(Line(signal.a, signal.b, color(signal.channels)))
    return signal_lines


def simulate(lvl, ctrl, vision, draw):
    draw.update(lvl, lines(vision.look()))
    while not (_done[0]):
        if _action[0] is None:
            time.sleep(1 / 60)
        else:
            ctrl.step(_action[0])
            _action[0] = None
            draw.update(lvl, lines(vision.look()))


def on_key_press(symbol, modifiers):
    if symbol == _KEY_WALK_FORWARD:
        _action[0] = controller.walk_forward
    elif symbol == _KEY_WALK_BACKWARD:
        _action[0] = controller.walk_backward
    elif symbol == _KEY_TURN_LEFT:
        _action[0] = controller.turn_left
    elif symbol == _KEY_TURN_RIGHT:
        _action[0] = controller.turn_right


def on_close():
    _done[0] = True


if __name__ == "__main__":
    main()
