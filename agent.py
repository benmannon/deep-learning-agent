import tensorflow as tf

_OP_INPUTS = 'inputs'
_OP_DROPOUT = 'dropout'
_OP_P = 'p'
_OP_GREEDY = 'greedy'
_OP_E_GREEDY = 'e_greedy'
_OP_EPSILON = 'e'
_OP_REWARDS = 'rewards'
_OP_ACTIONS = 'actions'
_OP_TRANSITIONS = 'transitions'
_OP_LEARNING_RATE = 'rate'
_OP_TRAIN = 'train'

_DROPOUT_OFF = 0.0
_DROPOUT_ON = 1.0


class Agent(object):
    def __init__(self, n_inputs, n_channels, n_outputs):
        self._sess, self._ops = self.build_model(n_inputs, n_channels, n_outputs)

    def build_model(self, n_inputs, n_channels, n_outputs):
        # feed forward
        x = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        dropout = tf.placeholder(tf.float32, [])
        q = self._model_q(x, dropout, n_inputs, n_channels, n_outputs)

        p = tf.nn.softmax(q)
        greedy = tf.argmax(q, 1)
        e = tf.placeholder(tf.float32, [])
        e_greedy = tf.select(tf.random_uniform(tf.shape(greedy)) < e,
                             tf.random_uniform(tf.shape(greedy), dtype=tf.int64, maxval=n_outputs),
                             greedy)

        # learning rate, actions, rewards, transitions
        rate = tf.placeholder(tf.float32, [])
        actions = tf.placeholder(tf.int32, [None])
        rewards = tf.placeholder(tf.float32, [None])
        transitions = tf.placeholder(tf.float32, [None, n_inputs, n_channels])

        # back propagation
        train = self._model_train(rate, p, actions, rewards, transitions)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        ops = {
            _OP_INPUTS: x,
            _OP_DROPOUT: dropout,
            _OP_P: p,
            _OP_GREEDY: greedy,
            _OP_E_GREEDY: e_greedy,
            _OP_EPSILON: e,
            _OP_REWARDS: rewards,
            _OP_ACTIONS: actions,
            _OP_TRANSITIONS: transitions,
            _OP_LEARNING_RATE: rate,
            _OP_TRAIN: train
        }

        return sess, ops

    def _model_train(self, rate, p, actions, rewards, transitions):
        # train is a no-op if there are no trainable variables
        if not tf.trainable_variables():
            return tf.no_op()

        # determine preferred action, calculate loss according to magnitude of reward
        action_flat = tf.range(0, tf.shape(p)[0]) * tf.shape(p)[1] + actions
        p_preferred = tf.sign(tf.sign(rewards) - 0.5) * 0.5 + 0.5
        p_action = tf.gather(tf.reshape(p, [-1]), action_flat)
        diff = p_preferred * (p_preferred - p_action) + (1 - p_preferred) * (p_preferred + p_action)
        log_diff = tf.log(tf.clip_by_value(diff, 1e-10, 1.0))
        loss = tf.reduce_mean(log_diff * tf.abs(rewards))

        # minimize loss
        train = tf.train.AdamOptimizer(rate).minimize(loss)

        return train

    def eval_pg(self, state):
        return self._sess.run(self._ops[_OP_P], feed_dict={
            self._ops[_OP_INPUTS]: [state],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
        })[0]

    def eval_greedy(self, state):
        return self._sess.run(self._ops[_OP_GREEDY], feed_dict={
            self._ops[_OP_INPUTS]: [state],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
        })[0]

    def eval_e_greedy(self, state, epsilon):
        return self._sess.run(self._ops[_OP_E_GREEDY], feed_dict={
            self._ops[_OP_INPUTS]: [state],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
            self._ops[_OP_EPSILON]: epsilon
        })[0]

    def eval_thompson_sample(self, state):
        return self._sess.run(self._ops[_OP_GREEDY], feed_dict={
            self._ops[_OP_INPUTS]: [state],
            self._ops[_OP_DROPOUT]: _DROPOUT_ON,
        })[0]

    def train(self, learning_rate, states, actions, rewards, states2):
        self._sess.run(self._ops[_OP_TRAIN], feed_dict={
            self._ops[_OP_LEARNING_RATE]: learning_rate,
            self._ops[_OP_INPUTS]: states,
            self._ops[_OP_ACTIONS]: actions,
            self._ops[_OP_REWARDS]: rewards,
            self._ops[_OP_DROPOUT]: _DROPOUT_ON,
            self._ops[_OP_TRANSITIONS]: states2
        })


class RandomAgent(Agent):
    def _model_q(self, x, n_inputs, n_channels, n_outputs):
        # ignore state, just be random
        q = tf.random_normal([tf.shape(x)[0], n_outputs])

        return q


class LinearAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class ReluAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class SigmoidAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.sigmoid,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q


class TanhAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.tanh,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1)
        )

        return q
