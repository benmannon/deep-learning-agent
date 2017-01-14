from __future__ import division

from agent import TanhAgent
from xp_buffer import XpBuffer


class Learner:
    def __init__(self, buffer_cap, batch_size, discount, learn_start_t, learn_interval, e_start, e_end, e_start_t, e_end_t, n_inputs, n_channels, n_actions, report_interval):
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []
        self._xp_buf = XpBuffer(buffer_cap)
        self._batch_size = batch_size
        self._gamma = discount
        self._learn_start_t = learn_start_t
        self._learn_interval = learn_interval
        self._e_start = e_start
        self._e_end = e_end
        self._e_start_t = e_start_t
        self._e_end_t = e_end_t
        self._agent = TanhAgent(n_inputs, n_channels, n_actions)
        self._report_interval = report_interval
        self._recent_state = None
        self._recent_action = None
        self._step = 0

    def _add_xp(self, state, action, reward):
        self._ep_states.append(state)
        self._ep_actions.append(action)
        self._ep_rewards.append(reward)

    def _end_episode(self):
        d_rewards = self._discount(self._ep_rewards)
        for i in range(0, len(self._ep_states)):
            s = self._ep_states[i]
            a = self._ep_actions[i]
            r = d_rewards[i]
            t = (i == len(self._ep_states) - 1)
            self._xp_buf.append(s, a, r, t)
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []

    def _discount(self, rewards):
        gamma = self._gamma
        total_reward = 0.0
        d_rewards = []
        for reward in reversed(rewards):
            total_reward = gamma * total_reward + reward
            d_rewards.append(total_reward)
        d_rewards.reverse()
        return d_rewards

    def _learn(self):
        if self._xp_buf.size > 0:
            states, actions, rewards, states2 = self._xp_buf.samples(self._batch_size)
            self._agent.train(states, actions, rewards, states2)

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

        if self._recent_state:
            self._add_xp(self._recent_state, self._recent_action, reward)

        epsilon = self._epsilon()
        action = self._agent.eval_e_greedy(state, epsilon)

        self._recent_state = state
        self._recent_action = action

        if terminal:
            self._end_episode()

        learning = self._step >= self._learn_start_t
        if learning and self._step % self._learn_interval == 0:
            self._learn()

        if self._step % self._report_interval == 0:
            print 'step=%s | {e: %s, learning: %s}' % (self._step, epsilon, learning)

        self._step += 1

        return action
