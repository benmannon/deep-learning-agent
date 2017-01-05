from xp_buffer import XpBuffer


class Learner:

    def __init__(self):
        self._episode = []
        self._xp_buf = XpBuffer(10000)

    def add_xp(self, xp):
        self._episode += [xp]

    def end_episode(self):
        self._xp_buf.append(self.discount(self._episode))
        self._episode = []

    def learn(self, agent):
        xp_samples = self._xp_buf.samples(1000)
        for xp_sample in xp_samples:
            state = xp_sample[0]
            action = xp_sample[1]
            reward = xp_sample[2]
            agent.train(state, action, reward, None)

    @staticmethod
    def discount(xps):
        # TODO calculate discounted rewards over time
        return xps
