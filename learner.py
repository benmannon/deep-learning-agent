from agent import TanhAgent
from xp_buffer import XpBuffer


class Learner:
    def __init__(self, buffer_size, batch_size, discount_factor, n_inputs, n_channels, n_actions):
        self._episode = []
        self._xp_buf = XpBuffer(buffer_size)
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._agent = TanhAgent(n_inputs, n_channels, n_actions)
        self._recent_state = None
        self._recent_action = None

    def _add_xp(self, xp):
        self._episode += [xp]

    def _end_episode(self):
        self._xp_buf.append(self._discount(self._episode))
        self._episode = []

    def _discount(self, xps):
        gamma = self._discount_factor
        total_reward = 0.0
        d_xps = []
        for xp in reversed(xps):
            total_reward = gamma * total_reward + xp[2]
            d_xps.append((xp[0], xp[1], total_reward))
        return reversed(d_xps)

    def _learn(self):
        if self._xp_buf.size > 0:
            xp_samples = self._xp_buf.samples(self._batch_size)
            states, actions, rewards = self._split(xp_samples)
            self._agent.train(states, actions, rewards)

    @staticmethod
    def _split(xps):
        states = []
        actions = []
        rewards = []
        for xp in xps:
            states.append(xp[0])
            actions.append(xp[1])
            rewards.append(xp[2])
        return states, actions, rewards

    def perceive(self, state, reward, terminal):

        if self._recent_state:
            self._add_xp((self._recent_state, self._recent_action, reward))

        action = self._agent.eval_e_greedy(state)

        self._recent_state = state
        self._recent_action = action

        if terminal:
            self._learn()

        return action
