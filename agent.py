import tensorflow as tf


class Agent(object):

    def __init__(self, n_inputs, n_channels, n_outputs):
        self._sess, self._x, self._p, self._reward, self._action, self._train = self.build_model(n_inputs, n_channels, n_outputs)

    def build_model(self, n_inputs, n_channels, n_outputs):

        # feed forward
        x = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        q = self._model_q(x, n_inputs, n_channels, n_outputs)
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
        return self._sess.run(self._p, feed_dict={self._x: [state]})[0]

    def train(self, states, actions, rewards):
        self._sess.run(self._train, feed_dict={
            self._x: states,
            self._action: actions,
            self._reward: rewards
        })


class RandomAgent(Agent):

    def _model_q(self, x, n_inputs, n_channels, n_outputs):

        # ignore state, just be random
        q = tf.random_normal([tf.shape(x)[0], n_outputs])

        return q


class LinearAgent(Agent):

    def _model_q(self, x, n_inputs, n_channels, n_outputs):

        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        q = tf.contrib.layers.fully_connected(
            inputs=x_flat,
            num_outputs=n_outputs,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class ReluAgent(Agent):

    def _model_q(self, x, n_inputs, n_channels, n_outputs):

        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        q = tf.contrib.layers.fully_connected(
            inputs=x_flat,
            num_outputs=n_outputs,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class SigmoidAgent(Agent):

    def _model_q(self, x, n_inputs, n_channels, n_outputs):

        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        q = tf.contrib.layers.fully_connected(
            inputs=x_flat,
            num_outputs=n_outputs,
            activation_fn=tf.sigmoid,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class TanhAgent(Agent):

    def _model_q(self, x, n_inputs, n_channels, n_outputs):

        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        q = tf.contrib.layers.fully_connected(
            inputs=x_flat,
            num_outputs=n_outputs,
            activation_fn=tf.tanh,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q
