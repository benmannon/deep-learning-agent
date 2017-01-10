import random

import numpy as np
import tensorflow as tf


def weights(shape):
    return tf.truncated_normal(shape, stddev=0.1)


def biases(shape):
    return tf.constant(0.1, shape=shape)


class Agent(object):
    def __init__(self, state_shape, action_range):
        self._state_shape = state_shape
        self._action_range = action_range

    def eval(self, state):
        pass

    def train(self, state, action, reward, rate):
        pass


class RandomAgent(Agent):
    def eval(self, state):
        # just be random
        return random.randrange(0, self._action_range)


class ShallowAgent(Agent):
    def __init__(self, state_shape, action_range):
        super(ShallowAgent, self).__init__(state_shape, action_range)
        self._state_size = np.prod(self._state_shape)
        self._sess, self._x, self._p, self._reward, self._action, self._train, self._loss = self.build_model()

    def build_model(self):
        # feed forward
        x = tf.placeholder(tf.float32, [None, self._state_size])
        w = tf.Variable(weights([self._state_size, self._action_range]))
        b = tf.Variable(biases([self._action_range]))
        y = tf.matmul(x, w) + b
        p = tf.nn.softmax(y)

        # back propagation
        reward = tf.placeholder(tf.float32, [None])
        action = tf.placeholder(tf.int32, [None])
        action_flat = tf.range(0, tf.shape(p)[0]) * tf.shape(p)[1] + action
        p_preferred = tf.sign(tf.sign(reward) - 0.5) * 0.5 + 0.5
        p_action = tf.gather(tf.reshape(p, [-1]), action_flat)
        diff = p_preferred * (p_preferred - p_action) + (1 - p_preferred) * (p_preferred + p_action)
        log_diff = tf.log(tf.clip_by_value(diff, 1e-10, 1.0))
        loss = tf.reduce_mean(log_diff * tf.abs(reward))
        train = tf.train.AdamOptimizer(0.05).minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess, x, p, reward, action, train, loss

    def eval(self, state):
        inputs = np.reshape(state, (1, self._state_size))
        return self._sess.run(self._p, feed_dict={self._x: inputs})[0]

    def train(self, states, actions, rewards):
        self._sess.run(self._train, feed_dict={
            self._x: np.reshape(states, (len(states), self._state_size)),
            self._action: actions,
            self._reward: rewards
        })
