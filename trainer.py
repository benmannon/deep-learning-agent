from __future__ import division

import getopt
import sys
import time
from math import pi
from threading import Lock

import controller
import simulator
from learner import Learner
from simulator import Simulator

# simulation parameters
_AGENT_VISION_RES = 16
_AGENT_VISION_FOV = pi / 2
_AGENT_VISION_ATTENUATION = 0.25
_AGENT_RADIUS = 0.45
_AGENT_STRIDE = 0.1
_AGENT_STRIDE_ON_TURN = 0.05
_AGENT_TURN = pi / 16
_COIN_RADIUS = 0.25
_REWARD_COIN = 1
_REWARD_WIN = 5
_REWARD_LOSS = -5
_REWARD_COLLISION = -0.1

# visualization parameters
_WINDOW_WIDTH = 2048
_GRID_COLOR = [0.8, 0.8, 0.8]
_COIN_COLOR = [1.0, 0.843, 0.0]
_AGENT_COLOR = [0.15, 0.45, 0.35]
_AGENT_POINTER_BRIGHTNESS = 0.5
_BKG_COLOR = [0.0, 0.3, 0.5]

# learning parameters
_REPLAY_BUFFER_SIZE = 1000000
_REPLAY_BATCH_SIZE = 32
_REWARD_DISCOUNT_FACTOR = 0.99
_LEARNING_RATE = 0.00025
_LEARN_START_T = 2000
_LEARN_INTERVAL = 4
_TARGET_UPDATE_INTERVAL = 200
_E_START = 1.0
_E_END = 0.1
_E_START_T = 2000
_E_END_T = 10000
_REPORT_INTERVAL = 1000

# user input (w, a, d)
_KEY_WALK_FORWARD = 87
_KEY_TURN_LEFT = 65
_KEY_TURN_RIGHT = 68


class Trainer:
    def __init__(self, model, user_control=False):
        self._model = model
        self._user_control = user_control
        self._action_lock = Lock()
        self._action = None
        self._done = False

    def train(self, sim, first_input):

        learner = Learner(buffer_cap=_REPLAY_BUFFER_SIZE,
                          batch_size=_REPLAY_BATCH_SIZE,
                          discount=_REWARD_DISCOUNT_FACTOR,
                          learning_rate=_LEARNING_RATE,
                          learn_start_t=_LEARN_START_T,
                          learn_interval=_LEARN_INTERVAL,
                          target_update_interval=_TARGET_UPDATE_INTERVAL,
                          e_start=_E_START,
                          e_end=_E_END,
                          e_start_t=_E_START_T,
                          e_end_t=_E_END_T,
                          model=self._model,
                          n_inputs=_AGENT_VISION_RES,
                          n_channels=simulator.CHANNEL_NUM,
                          n_actions=len(controller.actions),
                          report_interval=_REPORT_INTERVAL)

        agent_input = first_input
        reward = 0.0
        terminal = False

        while not self._done:

            if not self._user_control:
                action_i = learner.perceive(agent_input, reward, terminal)
                self._action = controller.actions[action_i]

            self._action_lock.acquire()
            try:
                action = self._action
                self._action = None
            finally:
                self._action_lock.release()

            if action is not None:
                agent_input, reward, terminal = sim.step(action)
            else:
                time.sleep(1 / 60)

    def key_press(self, symbol, modifiers):

        if not self._user_control:
            return

        self._action_lock.acquire()
        try:
            if symbol == _KEY_WALK_FORWARD:
                self._action = controller.walk_forward
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
        opts, args = getopt.getopt(argv, 'hu:m:', ['user-control', 'model'])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    # default arguments
    model = 'linear'
    user_control = False

    # apply arguments
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ('-u', '--user-control'):
            user_control = True
        elif opt in ('-m', '--model'):
            model = arg

    print "Model is '%s'" % model
    print 'User controls %s' % ('enabled' if user_control else 'disabled')

    trainer = Trainer(model, user_control=user_control)
    Simulator(agent_vision_res=_AGENT_VISION_RES,
              agent_vision_fov=_AGENT_VISION_FOV,
              agent_vision_attenuation=_AGENT_VISION_ATTENUATION,
              agent_radius=_AGENT_RADIUS,
              agent_stride=_AGENT_STRIDE,
              agent_stride_on_turn=_AGENT_STRIDE_ON_TURN,
              agent_turn=_AGENT_TURN,
              coin_radius=_COIN_RADIUS,
              reward_coin=_REWARD_COIN,
              reward_win=_REWARD_WIN,
              reward_loss=_REWARD_LOSS,
              reward_collision=_REWARD_COLLISION,
              window_width=_WINDOW_WIDTH,
              grid_color=_GRID_COLOR,
              coin_color=_COIN_COLOR,
              agent_color=_AGENT_COLOR,
              agent_pointer_brightness=_AGENT_POINTER_BRIGHTNESS,
              bkg_color=_BKG_COLOR) \
        .run(trainer.train, trainer.key_press, trainer.close)


def print_usage():
    print 'trainer.py [-u] [--user-control] [-m <model>] [--model <model>]'


if __name__ == "__main__":
    main(sys.argv[1:])
