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
        self._state_size = np.prod(self._state_shape)
        self._sess, self._x, self._p, self._reward, self._action, self._train = self.build_model()

    def build_model(self):

        # feed forward
        x = tf.placeholder(tf.float32, [None, self._state_size])
        q = self._model_q(x)
        p = tf.nn.softmax(q)

        # back propagation
        action, reward, train = self._model_train(p)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess, x, p, reward, action, train

    def _model_train(self, p):

        # action, reward
        reward = tf.placeholder(tf.float32, [None])
        action = tf.placeholder(tf.int32, [None])

        # train is a no-op if there are no trainable variables
        if not tf.trainable_variables():
            return action, reward, tf.no_op()

        # determine preferred action, calculate loss according to magnitude of reward
        action_flat = tf.range(0, tf.shape(p)[0]) * tf.shape(p)[1] + action
        p_preferred = tf.sign(tf.sign(reward) - 0.5) * 0.5 + 0.5
        p_action = tf.gather(tf.reshape(p, [-1]), action_flat)
        diff = p_preferred * (p_preferred - p_action) + (1 - p_preferred) * (p_preferred + p_action)
        log_diff = tf.log(tf.clip_by_value(diff, 1e-10, 1.0))
        loss = tf.reduce_mean(log_diff * tf.abs(reward))

        # minimize loss
        train = tf.train.AdamOptimizer(0.05).minimize(loss)

        return action, reward, train

    def eval(self, state):
        inputs = np.reshape(state, (1, self._state_size))
        return self._sess.run(self._p, feed_dict={self._x: inputs})[0]

    def train(self, states, actions, rewards):
        self._sess.run(self._train, feed_dict={
            self._x: np.reshape(states, (len(states), self._state_size)),
            self._action: actions,
            self._reward: rewards
        })


class RandomAgent(Agent):

    def _model_q(self, x):

        # ignore state, just be random
        q = tf.random_normal([tf.shape(x)[0], self._action_range])

        return q


class LinearAgent(Agent):

    def _model_q(self, x):

        # trainable variables
        w = tf.Variable(weights([self._state_size, self._action_range]))
        b = tf.Variable(biases([self._action_range]))

        # linear classifier
        q = tf.matmul(x, w) + b

        return q


class ReluAgent(Agent):

    def _model_q(self, x):

        # trainable variables
        w = tf.Variable(weights([self._state_size, self._action_range]))

        # fully connected layer
        q = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=self._action_range,
            variables_collections=[w],
            activation_fn=tf.nn.relu,
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class SigmoidAgent(Agent):

    def _model_q(self, x):

        # trainable variables
        w = tf.Variable(weights([self._state_size, self._action_range]))

        # fully connected layer
        q = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=self._action_range,
            variables_collections=[w],
            activation_fn=tf.sigmoid,
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class TanhAgent(Agent):

    def _model_q(self, x):

        # trainable variables
        w = tf.Variable(weights([self._state_size, self._action_range]))

        # fully connected layer
        q = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=self._action_range,
            variables_collections=[w],
            activation_fn=tf.tanh,
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q
