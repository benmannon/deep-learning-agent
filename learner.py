from xp_buffer import XpBuffer


class Learner:
    def __init__(self, buffer_size, batch_size, discount_factor):
        self._episode = []
        self._xp_buf = XpBuffer(buffer_size)
        self._batch_size = batch_size
        self._discount_factor = discount_factor

    def add_xp(self, xp):
        self._episode += [xp]

    def end_episode(self):
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

    def learn(self, agent):
        xp_samples = self._xp_buf.samples(self._batch_size)
        states, actions, rewards = self._split(xp_samples)
        agent.train(states, actions, rewards)

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
