from __future__ import division

import getopt
import sys
import time
from collections import namedtuple
from math import pi
from threading import Lock

import controller
import simulator
from learner import Learner
from simulator import Simulator

_ARGS_DICT = {
    # simulation parameters
    'agent_vision_res': 16,
    'agent_vision_fov': pi / 2,
    'agent_vision_attenuation': 0.25,
    'agent_radius': 0.45,
    'agent_stride': 0.1,
    'agent_stride_on_turn': 0.05,
    'agent_turn': pi / 16,
    'coin_radius': 0.25,
    'reward_coin': 1,
    'reward_win': 5,
    'reward_loss': -5,
    'reward_collision': -0.1,

    # visualization parameters
    'window_width': 2048,
    'grid_color': [0.8, 0.8, 0.8],
    'coin_color': [1.0, 0.843, 0.0],
    'agent_color': [0.15, 0.45, 0.35],
    'agent_pointer_brightness': 0.5,
    'bkg_color': [0.0, 0.3, 0.5],

    # learning parameters
    'replay_buffer_size': 1000000,
    'replay_batch_size': 32,
    'reward_discount_factor': 0.99,
    'learning_rate': 0.00025,
    'learn_start_t': 2000,
    'learn_interval': 4,
    'target_update_interval': 200,
    'e_start': 1.0,
    'e_end': 0.1,
    'e_start_t': 2000,
    'e_end_t': 10000,
    'report_interval': 1000,

    # user input (w, a, d)
    'key_walk_forward': 87,
    'key_turn_left': 65,
    'key_turn_right': 68
}

_ARGS = namedtuple('Args', _ARGS_DICT.keys())(**_ARGS_DICT)


class Trainer:
    def __init__(self, model, user_control=False):
        self._model = model
        self._user_control = user_control
        self._action_lock = Lock()
        self._action = None
        self._done = False

    def train(self, sim, first_input):

        learner = Learner(_ARGS, model=self._model, n_channels=simulator.CHANNEL_NUM, n_actions=len(controller.ACTIONS))

        agent_input = first_input
        reward = 0.0
        terminal = False

        while not self._done:

            if not self._user_control:
                action_i = learner.perceive(agent_input, reward, terminal)
                self._action = controller.ACTIONS[action_i]

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
            if symbol == _ARGS.key_walk_forward:
                self._action = controller.ACTION_WALK_FORWARD
            elif symbol == _ARGS.key_turn_left:
                self._action = controller.ACTION_TURN_LEFT
            elif symbol == _ARGS.key_turn_right:
                self._action = controller.ACTION_TURN_RIGHT
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
    Simulator(_ARGS).run(trainer.train, trainer.key_press, trainer.close)


def print_usage():
    print 'trainer.py [-u] [--user-control] [-m <model>] [--model <model>]'


if __name__ == "__main__":
    main(sys.argv[1:])
