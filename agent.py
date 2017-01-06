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
        self._sess, self._x, self._p, self._reward, self._action, self._train = self.build_model()

    def build_model(self):
        # feed forward
        x = tf.placeholder(tf.float32, [None, self._state_size])
        w = tf.Variable(weights([self._state_size, self._action_range]))
        b = tf.Variable(biases([self._action_range]))
        p = tf.reshape(tf.nn.softmax(tf.matmul(x, w) + b), (self._action_range,))

        # back propagation
        reward = tf.placeholder(tf.float32, [1])
        action = tf.placeholder(tf.int32, [1])
        p_action = tf.slice(p, action, [1])
        diff = 1 - p_action
        loss = diff * reward
        train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess, x, p, reward, action, train

    def eval(self, state):
        inputs = np.reshape(state, (1, self._state_size))
        return self._sess.run(self._p, feed_dict={self._x: inputs})

    def train(self, state, action, reward, rate):
        self._sess.run(self._train, feed_dict={
            self._x: np.reshape(state, (1, self._state_size)),
            self._action: [action],
            self._reward: [reward]
        })