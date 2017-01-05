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
        self._xp_buf.append(self.discount(self._episode))
        self._episode = []

    def learn(self, agent):
        xp_samples = self._xp_buf.samples(self._batch_size)
        for xp_sample in xp_samples:
            state = xp_sample[0]
            action = xp_sample[1]
            reward = xp_sample[2]
            agent.train(state, action, reward, None)

    def discount(self, xps):
        gamma = self._discount_factor
        total_reward = 0.0
        d_xps = []
        for xp in reversed(xps):
            total_reward = gamma * total_reward + xp[2]
            d_xps.append((xp[0], xp[1], total_reward))
        return reversed(d_xps)
