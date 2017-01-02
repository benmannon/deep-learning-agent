from __future__ import division

import time
from math import pi
from threading import Lock

import controller
from simulator import Simulator

# simulation parameters
_AGENT_VISION_RES = 32
_AGENT_VISION_FOV = pi / 2
_AGENT_VISION_ATTENUATION = 0.25
_AGENT_RADIUS = 0.45
_AGENT_STRIDE = 0.1
_AGENT_TURN = pi / 8
_COIN_RADIUS = 0.25

# visualization parameters
_WINDOW_SIZE = [2048, 1024]
_GRID_COLOR = [0.8, 0.8, 0.8]
_COIN_COLOR = [1.0, 0.843, 0.0]
_AGENT_COLOR = [0.15, 0.45, 0.35]
_AGENT_POINTER_BRIGHTNESS = 0.5
_BKG_COLOR = [0.0, 0.3, 0.5]

# user input (w, s, a, d)
_KEY_WALK_FORWARD = 87
_KEY_WALK_BACKWARD = 83
_KEY_TURN_LEFT = 65
_KEY_TURN_RIGHT = 68


class Trainer:
    def __init__(self):
        self._action_lock = Lock()
        self._action = None
        self._done = False

    def train(self, sim, first_input):
        while not self._done:
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
    Simulator(agent_vision_res=_AGENT_VISION_RES,
              agent_vision_fov=_AGENT_VISION_FOV,
              agent_vision_attenuation=_AGENT_VISION_ATTENUATION,
              agent_radius=_AGENT_RADIUS,
              agent_stride=_AGENT_STRIDE,
              agent_turn=_AGENT_TURN,
              coin_radius=_COIN_RADIUS,
              window_size=_WINDOW_SIZE,
              grid_color=_GRID_COLOR,
              coin_color=_COIN_COLOR,
              agent_color=_AGENT_COLOR,
              agent_pointer_brightness=_AGENT_POINTER_BRIGHTNESS,
              bkg_color=_BKG_COLOR) \
        .run(trainer.train, trainer.key_press, trainer.close)


if __name__ == "__main__":
    main()
