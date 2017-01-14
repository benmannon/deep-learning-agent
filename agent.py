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
_OP_TERMINAL = 'terminal'
_OP_GAMMA = 'gamma'
_OP_LEARNING_RATE = 'rate'
_OP_TRAIN = 'train'

_DROPOUT_OFF = 0.0
_DROPOUT_ON = 1.0


class Agent(object):
    def __init__(self, n_inputs, n_channels, n_outputs):
        self._sess, self._params, self._params_target, self._ops = self.build_model(n_inputs, n_channels, n_outputs)
        self._target = [None] * len(self._params)
        self.update_target()

    def build_model(self, n_inputs, n_channels, n_outputs):
        # feed forward
        x = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        dropout = tf.placeholder(tf.float32, [])
        q, params = self._model_q(x, dropout, n_inputs, n_channels, n_outputs)

        p = tf.nn.softmax(q)
        greedy = tf.argmax(q, 1)
        e = tf.placeholder(tf.float32, [])
        e_greedy = tf.select(tf.random_uniform(tf.shape(greedy)) < e,
                             tf.random_uniform(tf.shape(greedy), dtype=tf.int64, maxval=n_outputs),
                             greedy)

        # learning rate, actions, rewards, transitions
        gamma = tf.placeholder(tf.float32, [])
        rate = tf.placeholder(tf.float32, [])
        actions = tf.placeholder(tf.int32, [None])
        rewards = tf.placeholder(tf.float32, [None])
        x_target = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        terminals = tf.placeholder(tf.float32, [None])

        # target network, using target parameters
        q_target, params_target = self._model_q(x_target, _DROPOUT_OFF, n_inputs, n_channels, n_outputs, trainable=False)

        # back propagation
        train = self._model_train(gamma, rate, q, q_target, actions, rewards, terminals)

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
            _OP_TRANSITIONS: x_target,
            _OP_TERMINAL: terminals,
            _OP_GAMMA: gamma,
            _OP_LEARNING_RATE: rate,
            _OP_TRAIN: train
        }

        return sess, params, params_target, ops

    def _model_train(self, gamma, rate, q, q_target, actions, rewards, terminals):
        # train is a no-op if there are no trainable variables
        if not tf.trainable_variables():
            return tf.no_op()

        # q(s, a)
        q_flat = tf.range(0, tf.shape(q)[0]) * tf.shape(q)[1] + actions
        q_a = tf.gather(tf.reshape(q, [-1]), q_flat)

        # q_target = (1 - terminal) * gamma * Q2(s2, argmax_a Q2(s2, a))
        q2_a = tf.cast(tf.arg_max(q_target, 1), tf.int32)
        q2_flat = tf.range(0, tf.shape(q_target)[0]) * tf.shape(q_target)[1] + q2_a
        q2_max_a = tf.gather(tf.reshape(q_target, [-1]), q2_flat)
        q2 = (1 - terminals) * gamma * q2_max_a

        # loss = (reward + q_target(s', a') - q(s, a)) ^ 2
        loss = tf.pow(rewards + q2 - q_a, 2)
        loss_mean = tf.reduce_mean(loss)

        # minimize loss
        train = tf.train.RMSPropOptimizer(rate, momentum=0.95, epsilon=0.01).minimize(loss_mean)

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

    def train(self, discount, learning_rate, states, actions, rewards, states2, term2):
        feed_dict = {
            self._ops[_OP_GAMMA]: discount,
            self._ops[_OP_LEARNING_RATE]: learning_rate,
            self._ops[_OP_INPUTS]: states,
            self._ops[_OP_ACTIONS]: actions,
            self._ops[_OP_REWARDS]: rewards,
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
            self._ops[_OP_TRANSITIONS]: states2,
            self._ops[_OP_TERMINAL] : self.bools_to_floats(term2)
        }

        # feed parameters of current target network
        for i in range(0, len(self._target)):
            feed_dict[self._params_target[i]] = self._target[i]

        self._sess.run(self._ops[_OP_TRAIN], feed_dict=feed_dict)

    def bools_to_floats(self, bools):
        floats = [None] * len(bools)
        for i in range(0, len(bools)):
            floats[i] = 1.0 if bools[i] else 0.0
        return floats

    def update_target(self):
        for i in range(0, len(self._params)):
            self._target[i] = self._sess.run(self._params[i])


class RandomAgent(Agent):
    def _model_q(self, x, n_inputs, n_channels, n_outputs):
        # ignore state, just be random
        q = tf.random_normal([tf.shape(x)[0], n_outputs])

        return q


class LinearAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs, trainable=True):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        w_initial = tf.truncated_normal([x_size, n_outputs], stddev=0.1)
        w = tf.Variable(w_initial, trainable=trainable)

        b_initial = tf.constant(0.1, shape=[n_outputs])
        b = tf.Variable(b_initial, trainable=trainable)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=None,
            variables_collections={
                'weights': [w],
                'biases': [b]
            }
        )

        return q, [w, b]


class ReluAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs, trainable=True):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        w_initial = tf.truncated_normal([x_size, n_outputs], stddev=0.1)
        w = tf.Variable(w_initial, trainable=trainable)

        b_initial = tf.constant(0.1, shape=[n_outputs])
        b = tf.Variable(b_initial, trainable=trainable)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.nn.relu,
            variables_collections={
                'weights': [w],
                'biases': [b]
            }
        )

        return q, [w, b]


class SigmoidAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs, trainable=True):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        w_initial = tf.truncated_normal([x_size, n_outputs], stddev=0.1)
        w = tf.Variable(w_initial, trainable=trainable)

        b_initial = tf.constant(0.1, shape=[n_outputs])
        b = tf.Variable(b_initial, trainable=trainable)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.sigmoid,
            variables_collections={
                'weights': [w],
                'biases': [b]
            }
        )

        return q, [w, b]


class TanhAgent(Agent):
    def _model_q(self, x, dropout, n_inputs, n_channels, n_outputs, trainable=True):
        x_size = n_inputs * n_channels
        x_flat = tf.reshape(x, [-1, x_size])

        keep_prob = 1 - dropout * 0.5
        drop_x = tf.nn.dropout(x_flat, keep_prob)

        w_initial = tf.truncated_normal([x_size, n_outputs], stddev=0.1)
        w = tf.Variable(w_initial, trainable=trainable)

        b_initial = tf.constant(0.1, shape=[n_outputs])
        b = tf.Variable(b_initial, trainable=trainable)

        q = tf.contrib.layers.fully_connected(
            inputs=drop_x,
            num_outputs=n_outputs,
            activation_fn=tf.tanh,
            variables_collections={
                'weights': [w],
                'biases': [b]
            }
        )

        return q, [w, b]
