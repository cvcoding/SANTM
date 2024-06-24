import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from aux_code import rnn_ops
import random


def _binary_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    Based on http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html

    :param x: input tensor
    :return: y=round(x) with gradients defined by the identity mapping (y=x)
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)


class CustomLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        ''' t_max should be a float value corresponding to the longest possible
        time dependency in the input. '''
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        super(CustomLSTMCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

            x_size = x.get_shape().as_list()[1]

            inputs = x
            all_input = tf.concat([x, h], 1)
            num_gates = 2

            W_xh = tf.get_variable('W_xh',
                                   [x_size + self.num_units, num_gates * self.num_units])
            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init(self.t_max,
                                                               num_gates))

            W_hh = tf.get_variable('W_hh', [x_size, self.num_units])
            bias_hh = tf.get_variable('bias_hh', [self.num_units],
                                      initializer=chrono_init(self.t_max, 1))

            W_bina = tf.get_variable('W_bina', [x_size + self.num_units, 1])
            bias_bina = tf.get_variable('bias_bina', [1],
                                        initializer=chrono_init(self.t_max, 5))

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 2:
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                f, i = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)
                j = tf.nn.bias_add(tf.matmul(inputs, W_hh), bias_hh)

                gate_bina = tf.sigmoid(tf.nn.bias_add(tf.matmul(all_input, W_bina), bias_bina))
                # update_gate = _binary_round(gate_bina)
                update_gate = gate_bina

                new_c = (1. - update_gate) * (c*tf.sigmoid(f) + tf.sigmoid(i)*self._activation(j))
                # new_c = c*tf.sigmoid(f) + tf.sigmoid(i)*self._activation(j)
                # new_c = (1. - update_gate)*c + tf.sigmoid(i) * self._activation(j)

                # beta = 1
                # new_c = (1. - update_gate)*tf.sigmoid(f) * c + (1 - tf.sigmoid(f - beta)) * self._activation(j)

                # new_h = tf.sigmoid(k)*new_c
                # new_c = rnn_ops.layer_norm(new_c, name="new_c")
                new_h = new_c

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            new_h_bgate = tf.concat([new_h, update_gate], 1)

            return new_h_bgate, new_state


class ScdLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        self.randomnum = random.random()
        super(ScdLSTMCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, input_nextt_gate, output_ta_t, time, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

            time_f = tf.cast(time, tf.float32)
            mmm = self.randomnum * time_f
            random_index = tf.to_int32(mmm, name='ToInt32')

            def condition(index, output_pres_l):
                return tf.less(index, 1)  # time

            def body(index, output_pres_l):
                output_t = tuple(ta.read(random_index) for ta in output_ta_t)
                output_t = nest.pack_sequence_as(structure=output_t, flat_sequence=output_t)
                output_ta_temp = output_t[0]
                output_pres_l = output_pres_l.write(index, output_ta_temp)
                return index + 1, output_pres_l

            def f1():
                index = tf.constant(0)
                output_pres = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
                result = tf.while_loop(condition, body, loop_vars=[index, output_pres])
                last_time, last_out = result
                final_out = tf.squeeze(last_out.stack())
                batch_size = tf.shape(final_out)[0]
                last_out = tf.concat([tf.expand_dims(final_out, -1), tf.expand_dims(h, -1)], axis=2)
                weight_full1 = tf.get_variable('weight_full1', [1, 1], initializer=weight_initializer(1))  # +1e-9
                # weight_full1 = tf.get_variable('weight_full1', [1, 1])  # +1e-9

                weight_full2 = 1 - weight_full1  # +1e-9
                weight_full = tf.concat([weight_full1, weight_full2], axis=1)

                weight = tf.tile(weight_full, [batch_size, 1])

                final_out0 = tf.squeeze(tf.matmul(last_out, tf.expand_dims(weight, -1)))

                return final_out0

            def f2():
                return h

            h = tf.cond(time > 0, f1, f2)

            x_size = x.get_shape().as_list()[1]

            all_input = tf.concat([x, h], 1)
            num_gates = 3

            W_xh = tf.get_variable('W_xh',
                                   [x_size + self.num_units, num_gates * self.num_units])

            # W_ih = tf.get_variable('W_ih',
            #                        [self.num_units, self.num_units])
            # bias_ih = tf.get_variable('bias_ih', [self.num_units],
            #                        initializer=chrono_init_scd(self.t_max, 1))

            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init_scd(self.t_max,
                                                               num_gates))

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 3:
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                f, i, j = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)
                # i = tf.nn.bias_add(tf.matmul(c, W_ih), bias_ih)

                update_gate = tf.expand_dims(input_nextt_gate, 1)
                new_c = update_gate*(c*tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)) + (1. - update_gate)*c

                # beta = 1
                # new_c = update_gate * (tf.sigmoid(f) * c + (1 - tf.sigmoid(f - beta))
                # * self._activation(j))+ (1. - update_gate)*c
                new_h = new_c

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            return new_h, new_state


def chrono_init(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))

        t_random = random_ops.random_uniform([1], minval=1.0,
                                             maxval=t_max, dtype=dtype, seed=42)
        uni_vals2 = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_random, dtype=dtype,
                                                    seed=42))
        if num_gates == 2:
            bias_f = uni_vals2
            bias_co = -uni_vals2  # tf.zeros(num_units)
            return tf.concat([bias_f, bias_co], 0)

        elif num_gates == 3:
            bias_f = uni_vals
            bias_i = -uni_vals
            bias_j = tf.zeros(num_units)
            return tf.concat([bias_f, bias_i, bias_j], 0)

        elif num_gates == 4:
            bias_bina = tf.zeros(1)
            return bias_bina

        elif num_gates == 5:
            bias_bina = -tf.log(random_ops.random_uniform([1], minval=1.0,
                                      maxval=t_random, dtype=dtype, seed=42))
            # bias_bina = -tf.log(random_ops.random_uniform([1], minval=t_random,
            #                           maxval=t_max, dtype=dtype, seed=42)) 1e-5
            return bias_bina

        elif num_gates == 1:
            bias_j = tf.zeros(num_units)
            return bias_j

    return _initializer


def chrono_init_scd(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))

        if num_gates == 2:
            bias_f = uni_vals
            bias_j = tf.zeros(num_units)  # -uni_vals2
            return tf.concat([bias_f, bias_j], 0)

        elif num_gates == 3:
            bias_f = uni_vals
            bias_i = -uni_vals
            bias_j = tf.zeros(num_units)  # -uni_vals2
            return tf.concat([bias_f, bias_i, bias_j], 0)

        elif num_gates == 4:
            bias_bina = tf.zeros(1)
            return bias_bina

        elif num_gates == 5:

            bias_bina = -tf.log(random_ops.random_uniform([1], minval=1.0,
                                      maxval=t_random, dtype=dtype, seed=42))
            return bias_bina

        elif num_gates == 1:
            bias_i = -uni_vals
            return bias_i

    return _initializer


def bias_initializer(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        p = np.zeros(shape)
        num_units = int(shape[0]//num_gates)
        # i, j, o, f
        # f:
        p[-num_units:] = np.ones(num_units)
        return tf.constant(p, dtype)

    return _initializer


def weight_initializer(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        # p = np.zeros(shape)
        # num_units = int(shape[0]//num_gates)
        # p[-num_units:] = np.ones(num_units)
        p = np.ones([1, 1])

        return tf.constant(p, dtype)

    return _initializer