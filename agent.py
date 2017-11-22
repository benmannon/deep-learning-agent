import tensorflow as tf

_OP_STATES_VISION = 'states_vision'
_OP_STATES_PREV_ACTION = 'states_prev_action'
_OP_DROPOUT = 'dropout'
_OP_P = 'p'
_OP_GREEDY = 'greedy'
_OP_E_GREEDY = 'e_greedy'
_OP_EPSILON = 'e'
_OP_REWARDS = 'rewards'
_OP_ACTIONS = 'actions'
_OP_STATES2_VISION = 'states2_vision'
_OP_STATES2_PREV_ACTION = 'states2_prev_action'
_OP_TERMINAL = 'terminal'
_OP_GAMMA = 'gamma'
_OP_LEARNING_RATE = 'rate'
_OP_TRAIN = 'train'

_PARAMS_Q = 'q'
_PARAMS_Q2 = 'q2'

_DROPOUT_OFF = 0.0
_DROPOUT_ON = 1.0


def _q_random(sv, prev_action, dropout, n_inputs, n_channels, n_outputs, trainable=True, params=None):
    # ignore state, just be random
    q = tf.random_normal([tf.shape(sv)[0], n_outputs])

    return q, []


def _q_fully_connected(activation_fn):
    def fn(sv, prev_action, dropout, n_inputs, n_channels, n_outputs, trainable=True, params=None):

        # flatten input
        sv_size = n_inputs * n_channels
        sv_flat = tf.reshape(sv, [-1, sv_size])

        # create and concatenate a one-hot previous-action vector
        a = tf.one_hot(prev_action, n_outputs)
        s = tf.concat([sv_flat, a], 1)

        # dropout will be 1.0 (ON) or 0.0 (OFF)
        keep_prob = 1 - dropout * 0.5
        drop_s = tf.nn.dropout(s, keep_prob)

        # parameters may be shared with an identical model
        if params is None:
            w_initial = tf.truncated_normal([sv_size, n_outputs], stddev=0.1)
            w = tf.Variable(w_initial, trainable=trainable)

            b_initial = tf.constant(0.1, shape=[n_outputs])
            b = tf.Variable(b_initial, trainable=trainable)

        else:
            w, b = params

        q = tf.contrib.layers.fully_connected(
            inputs=drop_s,
            num_outputs=n_outputs,
            activation_fn=activation_fn,
            variables_collections={
                'weights': [w],
                'biases': [b]
            }
        )

        return q, [w, b]

    return fn


def _q_hidden_fully_connected(activation_fn, n_hidden):
    def fn(sv, prev_action, dropout, n_inputs, n_channels, n_outputs, trainable=True, params=None):

        # flatten input
        sv_size = n_inputs * n_channels
        sv_flat = tf.reshape(sv, [-1, sv_size])

        # create and concatenate a one-hot previous-action vector
        a = tf.one_hot(prev_action, n_outputs)
        s = tf.concat([sv_flat, a], 1)

        # dropout will be 1.0 (ON) or 0.0 (OFF)
        keep_prob = 1 - dropout * 0.5
        drop_s = tf.nn.dropout(s, keep_prob)

        # parameters may be shared with an identical model
        if params is None:
            h_w_initial = tf.truncated_normal([sv_size, n_hidden], stddev=0.1)
            h_w = tf.Variable(h_w_initial, trainable=trainable)

            h_b_initial = tf.constant(0.1, shape=[n_hidden])
            h_b = tf.Variable(h_b_initial, trainable=trainable)

            q_w_initial = tf.truncated_normal([n_hidden, n_outputs], stddev=0.1)
            q_w = tf.Variable(q_w_initial, trainable=trainable)

            q_b_initial = tf.constant(0.1, shape=[n_outputs])
            q_b = tf.Variable(q_b_initial, trainable=trainable)

        else:
            h_w, h_b, q_w, q_b = params

        h = tf.contrib.layers.fully_connected(
            inputs=drop_s,
            num_outputs=n_hidden,
            activation_fn=activation_fn,
            variables_collections={
                'weights': [h_w],
                'biases': [h_b]
            }
        )

        q = tf.contrib.layers.fully_connected(
            inputs=h,
            num_outputs=n_outputs,
            activation_fn=activation_fn,
            variables_collections={
                'weights': [q_w],
                'biases': [q_b]
            }
        )

        return q, [h_w, h_b, q_w, q_b]

    return fn

q_models = {
    'random': _q_random,
    'linear': _q_fully_connected(None),
    'relu': _q_fully_connected(tf.nn.relu),
    'sigmoid': _q_fully_connected(tf.sigmoid),
    'tanh': _q_fully_connected(tf.tanh),
    'hidden_linear': _q_hidden_fully_connected(None, 2),
    'hidden_relu': _q_hidden_fully_connected(tf.nn.relu, 3),
    'hidden_sigmoid': _q_hidden_fully_connected(tf.sigmoid, 2),
    'hidden_tanh': _q_hidden_fully_connected(tf.tanh, 2)
}


def _tf_select(tensor, indices):
    offset = tf.range(0, tf.shape(tensor)[0]) * tf.shape(tensor)[1] + indices
    flat = tf.reshape(tensor, [-1])
    return tf.gather(flat, offset)


def _model_train(gamma, rate, q_s, q_s2, q2_s2, a, r, term):
    # train is a no-op if there are no trainable variables
    if not tf.trainable_variables():
        return tf.no_op()

    # Q(s, a)
    q_s_a = _tf_select(q_s, a)

    # a2 = argmax_a2 Q(s2, a2)
    a2 = tf.cast(tf.argmax(q_s2, 1), tf.int32)

    # Q2(s2, a2)
    q2_s2_a2 = _tf_select(q2_s2, a2)

    # Yt = reward + gamma * Q2(s2, a2)
    # (don't reward future terminal states)
    target = r + (1 - term) * gamma * q2_s2_a2

    # don't backpropagate into the target graph
    target_stop = tf.stop_gradient(target)

    # loss = (target - Q(s, a)) ^ 2
    loss = tf.pow(target_stop - q_s_a, 2)
    loss_mean = tf.reduce_mean(loss)

    # minimize loss
    train = tf.train.RMSPropOptimizer(rate, momentum=0.95, epsilon=0.01).minimize(loss_mean)

    return train


def _build_model(q_model, n_inputs, n_channels, n_outputs):
    # feed forward
    sv = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
    sa = tf.placeholder(tf.int32, [None])
    dropout = tf.placeholder(tf.float32, [])
    q_s, q_params = q_model(sv, sa, dropout, n_inputs, n_channels, n_outputs)

    # variety of evaluation functions
    p = tf.nn.softmax(q_s)
    greedy = tf.argmax(q_s, 1)
    e = tf.placeholder(tf.float32, [])
    e_greedy = tf.where(tf.random_uniform(tf.shape(greedy)) < e,
                        tf.random_uniform(tf.shape(greedy), dtype=tf.int64, maxval=n_outputs),
                        greedy)

    # learning rate, actions, rewards, transitions
    gamma = tf.placeholder(tf.float32, [])
    rate = tf.placeholder(tf.float32, [])
    a = tf.placeholder(tf.int32, [None])
    r = tf.placeholder(tf.float32, [None])
    s2v = tf.placeholder(tf.float32, [None, n_inputs, n_channels])
    s2a = tf.placeholder(tf.int32, [None])
    term = tf.placeholder(tf.float32, [None])

    # current network's evaluation of transition state
    q_s2, _ = q_model(s2v, s2a, _DROPOUT_OFF, n_inputs, n_channels, n_outputs, params=q_params)

    # target network's evaluation of transition state
    q2_s2, q2_params = q_model(s2v, s2a, _DROPOUT_OFF, n_inputs, n_channels, n_outputs, trainable=False)

    # back propagation
    train = _model_train(
        gamma=gamma,
        rate=rate,
        q_s=q_s,
        q_s2=q_s2,
        q2_s2=q2_s2,
        a=a,
        r=r,
        term=term
    )

    ops = {
        _OP_STATES_VISION: sv,
        _OP_STATES_PREV_ACTION: sa,
        _OP_DROPOUT: dropout,
        _OP_P: p,
        _OP_GREEDY: greedy,
        _OP_E_GREEDY: e_greedy,
        _OP_EPSILON: e,
        _OP_REWARDS: r,
        _OP_ACTIONS: a,
        _OP_STATES2_VISION: s2v,
        _OP_STATES2_PREV_ACTION: s2a,
        _OP_TERMINAL: term,
        _OP_GAMMA: gamma,
        _OP_LEARNING_RATE: rate,
        _OP_TRAIN: train
    }

    params = {
        _PARAMS_Q: q_params,
        _PARAMS_Q2: q2_params,
    }

    return ops, params


def _tf_init():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess


def _bools_to_floats(bools):
    floats = [None] * len(bools)
    for i in range(0, len(bools)):
        floats[i] = 1.0 if bools[i] else 0.0
    return floats


class Agent:
    def __init__(self, q_model, n_inputs, n_channels, n_outputs):
        self._ops, self._params = _build_model(q_model, n_inputs, n_channels, n_outputs)
        self._sess = _tf_init()
        self._target_params = [None] * len(self._params[_PARAMS_Q])
        self.update_target()

    def eval_pg(self, state):
        return self._sess.run(self._ops[_OP_P], feed_dict={
            self._ops[_OP_STATES_VISION]: [state.vision],
            self._ops[_OP_STATES_PREV_ACTION]: [state.prev_action],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
        })[0]

    def eval_greedy(self, state):
        return self._sess.run(self._ops[_OP_GREEDY], feed_dict={
            self._ops[_OP_STATES_VISION]: [state.vision],
            self._ops[_OP_STATES_PREV_ACTION]: [state.prev_action],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
        })[0]

    def eval_e_greedy(self, state, epsilon):
        return self._sess.run(self._ops[_OP_E_GREEDY], feed_dict={
            self._ops[_OP_STATES_VISION]: [state.vision],
            self._ops[_OP_STATES_PREV_ACTION]: [state.prev_action],
            self._ops[_OP_DROPOUT]: _DROPOUT_OFF,
            self._ops[_OP_EPSILON]: epsilon
        })[0]

    def eval_thompson_sample(self, state):
        return self._sess.run(self._ops[_OP_GREEDY], feed_dict={
            self._ops[_OP_STATES_VISION]: [state.vision],
            self._ops[_OP_STATES_PREV_ACTION]: [state.prev_action],
            self._ops[_OP_DROPOUT]: _DROPOUT_ON,
        })[0]

    def train(self, discount, learning_rate, states, actions, rewards, states2, term2):
        feed_dict = {
            self._ops[_OP_GAMMA]: discount,
            self._ops[_OP_LEARNING_RATE]: learning_rate,
            self._ops[_OP_STATES_VISION]: map(lambda state: state.vision, states),
            self._ops[_OP_STATES_PREV_ACTION]: map(lambda state: state.prev_action, states),
            self._ops[_OP_ACTIONS]: actions,
            self._ops[_OP_REWARDS]: rewards,
            self._ops[_OP_DROPOUT]: _DROPOUT_ON,
            self._ops[_OP_STATES2_VISION]: map(lambda state: state.vision, states2),
            self._ops[_OP_STATES2_PREV_ACTION]: map(lambda state: state.prev_action, states2),
            self._ops[_OP_TERMINAL]: _bools_to_floats(term2)
        }

        # feed parameters of current target network into Q2
        q2_params = self._params[_PARAMS_Q2]
        for i in range(0, len(self._target_params)):
            feed_dict[q2_params[i]] = self._target_params[i]

        self._sess.run(self._ops[_OP_TRAIN], feed_dict=feed_dict)

    def update_target(self):
        q_params = self._params[_PARAMS_Q]
        for i in range(0, len(self._target_params)):
            self._target_params[i] = self._sess.run(q_params[i])
