import random

import numpy as np
import tensorflow as tf


def weights(shape):
    return tf.truncated_normal(shape, stddev=0.1)


def biases(shape):
    return tf.constant(0.1, shape=shape)


class Agent(object):
    def __init__(self, input_shape, output_range):
        self._input_shape = input_shape
        self._output_range = output_range

    def eval(self, agent_input):
        pass


class RandomAgent(Agent):
    def eval(self, agent_input):
        # just be random
        return random.randrange(0, self._output_range)


class ShallowAgent(Agent):
    def __init__(self, input_shape, output_range):
        super(ShallowAgent, self).__init__(input_shape, output_range)
        self._input_size = np.prod(self._input_shape)
        self._sess, self._x, self._q_max = self.build_model()

    def build_model(self):
        x = tf.placeholder(tf.float32, [None, self._input_size])
        w = tf.Variable(weights([self._input_size, self._output_range]))
        b = tf.Variable(biases([self._output_range]))
        q = tf.nn.softmax(tf.matmul(x, w) + b)
        q_max = tf.argmax(q, 1)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess, x, q_max

    def eval(self, agent_input):
        inputs = np.reshape(agent_input, (1, self._input_size))
        return self._sess.run(self._q_max, feed_dict={self._x: inputs})
