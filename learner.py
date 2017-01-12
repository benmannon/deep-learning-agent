from agent import TanhAgent
from xp_buffer import XpBuffer


class Learner:
    def __init__(self, buffer_size, batch_size, discount_factor, n_inputs, n_channels, n_actions):
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []
        self._xp_buf = XpBuffer(buffer_size)
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._agent = TanhAgent(n_inputs, n_channels, n_actions)
        self._recent_state = None
        self._recent_action = None

    def _add_xp(self, state, action, reward):
        self._ep_states.append(state)
        self._ep_actions.append(action)
        self._ep_rewards.append(reward)

    def _end_episode(self):
        self._xp_buf.append(self._ep_states, self._ep_actions, self._discount(self._ep_rewards))
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []

    def _discount(self, rewards):
        gamma = self._discount_factor
        total_reward = 0.0
        d_rewards = []
        for reward in reversed(rewards):
            total_reward = gamma * total_reward + reward
            d_rewards.append(total_reward)
        return reversed(d_rewards)

    def _learn(self):
        if self._xp_buf.size > 0:
            states, actions, rewards = self._xp_buf.samples(self._batch_size)
            self._agent.train(states, actions, rewards)

    def perceive(self, state, reward, terminal):

        if self._recent_state:
            self._add_xp(self._recent_state, self._recent_action, reward)

        action = self._agent.eval_e_greedy(state)

        self._recent_state = state
        self._recent_action = action

        if terminal:
            self._end_episode()
            self._learn()

        return action
