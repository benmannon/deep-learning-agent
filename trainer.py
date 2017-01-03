from __future__ import division

import getopt
import time
from math import pi
from threading import Lock

import sys

import controller
from agent import RandomAgent
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
_WINDOW_WIDTH = 2048
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
    def __init__(self, user_control=False):
        self._user_control = user_control
        self._action_lock = Lock()
        self._action = None
        self._done = False
        self._agent = RandomAgent(_AGENT_VISION_RES, len(controller.actions))

    def train(self, sim, first_input):

        agent_input = first_input

        while not self._done:

            if not self._user_control:
                action_i = self._agent.eval(agent_input)
                self._action = controller.actions[action_i]

            self._action_lock.acquire()
            try:
                action = self._action
                self._action = None
            finally:
                self._action_lock.release()

            if action is not None:
                agent_input = sim.step(action)
            else:
                time.sleep(1 / 60)

    def key_press(self, symbol, modifiers):

        if not self._user_control:
            return

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


def main(argv):

    # parse command-line arguments
    try:
        opts, args = getopt.getopt(argv, 'hu:', ["user-control"])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    # apply arguments
    user_control = False
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ('-u', '--user-control'):
            user_control = True
            print('User controls enabled')

    trainer = Trainer(user_control=user_control)
    Simulator(agent_vision_res=_AGENT_VISION_RES,
              agent_vision_fov=_AGENT_VISION_FOV,
              agent_vision_attenuation=_AGENT_VISION_ATTENUATION,
              agent_radius=_AGENT_RADIUS,
              agent_stride=_AGENT_STRIDE,
              agent_turn=_AGENT_TURN,
              coin_radius=_COIN_RADIUS,
              window_width=_WINDOW_WIDTH,
              grid_color=_GRID_COLOR,
              coin_color=_COIN_COLOR,
              agent_color=_AGENT_COLOR,
              agent_pointer_brightness=_AGENT_POINTER_BRIGHTNESS,
              bkg_color=_BKG_COLOR) \
        .run(trainer.train, trainer.key_press, trainer.close)


def print_usage():
    print 'trainer.py [-u] [--user-control]'


if __name__ == "__main__":
    main(sys.argv[1:])
