from __future__ import division

import time
from threading import Lock

import controller
from simulator import Simulator

_KEY_WALK_FORWARD = 87  # w
_KEY_WALK_BACKWARD = 83  # s
_KEY_TURN_LEFT = 65  # a
_KEY_TURN_RIGHT = 68  # d


class Trainer:
    def __init__(self):
        self._action_lock = Lock()
        self._action = None
        self._done = False

    def train(self, sim, first_input):
        # track fps
        timer = time.time()
        frames = 0

        while not self._done:

            # print fps
            frames += 1
            time_delta = time.time() - timer
            fps_interval = 10.0
            if time_delta >= fps_interval:
                timer += fps_interval
                print 'fps over %ss: %s' % (fps_interval, frames / time_delta)
                frames = 0

            self._action_lock.acquire()
            try:
                action = self._action
                self._action = None
            finally:
                self._action_lock.release()

            if action is not None:
                sim.step(action)
            else:
                time.sleep(1 / 60)

    def key_press(self, symbol, modifiers):
        self._action_lock.acquire()
        try:
            if symbol == _KEY_WALK_FORWARD:
                self._action = controller.walk_forward
            elif symbol == _KEY_WALK_BACKWARD:
                self._action = controller.walk_backward
            elif symbol == _KEY_TURN_LEFT:
                self._action = controller.turn_left
            elif symbol == _KEY_TURN_RIGHT:
                self._action = controller.turn_right
        finally:
            self._action_lock.release()

    def close(self):
        self._done = True


def main():
    trainer = Trainer()
    Simulator().run(trainer.train, trainer.key_press, trainer.close)


if __name__ == "__main__":
    main()
