from __future__ import division

import time

import controller
from simulator import Simulator

_KEY_WALK_FORWARD = 87  # w
_KEY_WALK_BACKWARD = 83  # s
_KEY_TURN_LEFT = 65  # a
_KEY_TURN_RIGHT = 68  # d

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
