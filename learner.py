from __future__ import division

import random
import time
from agent import Agent, q_models
from xp_buffer import XpBuffer


class Learner:
    def __init__(self, args, model, n_channels, n_actions):
        self._xp_buf = XpBuffer(args.replay_buffer_size)
        self._batch_size = args.replay_batch_size
        self._gamma = args.reward_discount_factor
        self._learning_rate = args.learning_rate
        self._learn_start_t = args.learn_start_t
        self._learn_interval = args.learn_interval
        self._target_update_interval = args.target_update_interval
        self._e_start = args.e_start
        self._e_end = args.e_end
        self._e_start_t = args.e_start_t
        self._e_end_t = args.e_end_t
        self._agent = Agent(q_models[model], args.agent_vision_res, n_channels, n_actions)
        self._report_interval = args.report_interval
        self._recent_state = None
        self._recent_action = None
        self._step = 0
        self._timer = None

    def _add_xp(self, state, action, reward, term):
        self._xp_buf.append(state, action, reward, term)

    def _learn(self):
        if self._xp_buf.size > 0:
            states, actions, rewards, states2, term2 = self._xp_buf.samples(self._batch_size)
            self._agent.train(self._gamma, self._learning_rate, states, actions, rewards, states2, term2)

    def _epsilon(self):

        step = self._step
        e_start = self._e_start
        e_end = self._e_end
        e_start_t = self._e_start_t
        e_end_t = self._e_end_t

        # assume some constraints
        assert e_start >= e_end
        assert e_start_t <= e_end_t

        # linear annealing
        t = (step - e_start_t) / (e_end_t - e_start_t)
        e = e_start + t * (e_end - e_start)
        e = min(e, e_start)
        e = max(e, e_end)

        return e

    def perceive(self, state, reward, terminal):

        if self._timer is None:
            self._timer = time.clock()

        if self._recent_state:
            self._add_xp(self._recent_state, self._recent_action, reward, terminal)

        epsilon = self._epsilon()
        action = self._agent.eval_e_greedy(state, epsilon)

        self._recent_state = state
        self._recent_action = action

        learning = self._step >= self._learn_start_t
        if learning and (self._step + self._learn_start_t) % self._target_update_interval == 0:
            self._agent.update_target()

        if learning and self._step % self._learn_interval == 0:
            self._learn()

        if self._step % self._report_interval == 0:
            now = time.clock()
            elapsed = now - self._timer
            pace = self._report_interval / elapsed
            pace_str = 'na' if self._step == 0 else repr(round(pace, 1))
            self._timer = now
            print 'step=%s, pace=%s | {e: %s, learning: %s}'\
                  % (self._step, pace_str, epsilon, learning)

        self._step += 1

        return action

    @staticmethod
    def _select(p):

        # randomly select an index over an array of normalized probabilities
        r = random.random()
        odds = 0.0
        i = 0
        for prob in p:
            odds += prob
            if r <= odds:
                return i
            i += 1

        # should never get this far, but return the last item just in case
        print 'warning: total odds, %s < 1.0' % odds
        return i - 1
