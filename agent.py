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
        self._sess, self._q_params, self._q2_params, self._ops = self.build_model(n_inputs, n_channels, n_outputs)
        self._target_params = [None] * len(self._q_params)
        self.update_target()

    def build_model(self, n_inputs, n_channels, n_outputs):
        # feed forward
        s = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        dropout = tf.placeholder(tf.float32, [])
        q_s, q_params = self._model_q(s, dropout, n_inputs, n_channels, n_outputs)

        # variety of evaluation functions
        p = tf.nn.softmax(q_s)
        greedy = tf.argmax(q_s, 1)
        e = tf.placeholder(tf.float32, [])
        e_greedy = tf.select(tf.random_uniform(tf.shape(greedy)) < e,
                             tf.random_uniform(tf.shape(greedy), dtype=tf.int64, maxval=n_outputs),
                             greedy)

        # learning rate, actions, rewards, transitions
        gamma = tf.placeholder(tf.float32, [])
        rate = tf.placeholder(tf.float32, [])
        a = tf.placeholder(tf.int32, [None])
        r = tf.placeholder(tf.float32, [None])
        s2 = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
        term = tf.placeholder(tf.float32, [None])

        # target network, using target parameters
        q2_s2, q2_params = self._model_q(s2, _DROPOUT_OFF, n_inputs, n_channels, n_outputs, trainable=False)

        # back propagation
        train = self._model_train(
            gamma=gamma,
            rate=rate,
            q_s=q_s,
            q2_s2=q2_s2,
            a=a,
            r=r,
            term=term
        )

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        ops = {
            _OP_INPUTS: s,
            _OP_DROPOUT: dropout,
            _OP_P: p,
            _OP_GREEDY: greedy,
            _OP_E_GREEDY: e_greedy,
            _OP_EPSILON: e,
            _OP_REWARDS: r,
            _OP_ACTIONS: a,
            _OP_TRANSITIONS: s2,
            _OP_TERMINAL: term,
            _OP_GAMMA: gamma,
            _OP_LEARNING_RATE: rate,
            _OP_TRAIN: train
        }

        return sess, q_params, q2_params, ops

    def _model_train(self, gamma, rate, q_s, q2_s2, a, r, term):
        # train is a no-op if there are no trainable variables
        if not tf.trainable_variables():
            return tf.no_op()

        # Q(s, a)
        q_flat = tf.range(0, tf.shape(q_s)[0]) * tf.shape(q_s)[1] + a
        q_a = tf.gather(tf.reshape(q_s, [-1]), q_flat)

        # q2_a = argmax_a2 Q2(s2, a2)
        q2_a = tf.cast(tf.arg_max(q2_s2, 1), tf.int32)
        q2_a_flat = tf.range(0, tf.shape(q2_s2)[0]) * tf.shape(q2_s2)[1] + q2_a

        # Q2(s2, q2_a)
        q2_max_a = tf.gather(tf.reshape(q2_s2, [-1]), q2_a_flat)

        # Yt = reward + gamma * Q2(s2, q2_a)
        # (don't reward terminal states)
        target = r + (1 - term) * gamma * q2_max_a

        # loss = (target - q(s, a)) ^ 2
        loss = tf.pow(target - q_a, 2)
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

        # feed parameters of current target network into Q2
        for i in range(0, len(self._target_params)):
            feed_dict[self._q2_params[i]] = self._target_params[i]

        self._sess.run(self._ops[_OP_TRAIN], feed_dict=feed_dict)

    def bools_to_floats(self, bools):
        floats = [None] * len(bools)
        for i in range(0, len(bools)):
            floats[i] = 1.0 if bools[i] else 0.0
        return floats

    def update_target(self):
        for i in range(0, len(self._q_params)):
            self._target_params[i] = self._sess.run(self._q_params[i])


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
