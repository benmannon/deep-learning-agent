import tensorflow as tf

_OP_INPUTS = 'inputs'
_OP_P = 'p'
_OP_GREEDY = 'greedy'
_OP_E_GREEDY = 'e_greedy'
_OP_EPSILON = 'e'
_OP_REWARDS = 'rewards'
_OP_ACTIONS = 'actions'
_OP_TRAIN = 'train'


class Agent(object):
    def __init__(self, n_inputs, n_channels, n_outputs):
        self._sess, self._ops = self.build_model(n_inputs, n_channels, n_outputs)

    def build_model(self, n_inputs, n_channels, n_outputs):
        # feed forward
        x = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        q = self._model_q(x, n_inputs, n_channels, n_outputs)

        p = tf.nn.softmax(q)
        greedy = tf.argmax(q, 1)
        e = tf.placeholder(tf.float32, [])
        e_greedy = tf.select(tf.random_uniform(tf.shape(greedy)) < e,
                             tf.random_uniform(tf.shape(greedy), dtype=tf.int64, maxval=n_outputs),
                             greedy)

        # back propagation
        action, reward, train = self._model_train(p)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        ops = {
            _OP_INPUTS: x,
            _OP_P: p,
            _OP_GREEDY: greedy,
            _OP_E_GREEDY: e_greedy,
            _OP_EPSILON: e,
            _OP_REWARDS: reward,
            _OP_ACTIONS: action,
            _OP_TRAIN: train
        }

        return sess, ops

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

    def eval_pg(self, state):
        return self._sess.run(self._ops[_OP_P], feed_dict={self._ops[_OP_INPUTS]: [state]})[0]

    def eval_greedy(self, state):
        return self._sess.run(self._ops[_OP_GREEDY], feed_dict={self._ops[_OP_INPUTS]: [state]})[0]

    def eval_e_greedy(self, state, epsilon):
        return self._sess.run(self._ops[_OP_E_GREEDY], feed_dict={
            self._ops[_OP_INPUTS]: [state],
            self._ops[_OP_EPSILON]: epsilon
        })[0]

    def train(self, states, actions, rewards):
        self._sess.run(self._ops[_OP_TRAIN], feed_dict={
            self._ops[_OP_INPUTS]: states,
            self._ops[_OP_ACTIONS]: actions,
            self._ops[_OP_REWARDS]: rewards
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
