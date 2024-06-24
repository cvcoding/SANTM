import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
import collections
from utils import expand, learned_init, create_linear_initializer
import rnn_cell_impl_local_scd, rnn_cell_impl_doublelocal
import math


def init_u1(t_max):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        bias_j = 1 * tf.ones([1, t_max])
        return bias_j

    return _initializer


def init_D1(t_max):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        bias_j = 1 * tf.ones([1, t_max])
        return bias_j
    return _initializer


def _binary_round(x):
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
            batch_size = x.get_shape().as_list()[0]

            all_input = tf.concat([x, h], 1)
            all_inputc = tf.concat([x, c], 1)
            num_gates = 4

            W_xh = tf.get_variable('W_xh',
                                   [x_size + self.num_units, num_gates * self.num_units])
            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init_1long(self.t_max,
                                                               num_gates))
            W_bina = tf.get_variable('W_bina', [x_size + self.num_units, 1])
            bias_bina = tf.get_variable('bias_bina', [1],
                                        initializer=bias_initializer_reset(1))

            # u1 = tf.get_variable('u1', [1], initializer=init_u1(1))
            # D1 = tf.get_variable('D1', [1], initializer=init_D1(1))

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 4:  # divide c state
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)

                gate_bina = tf.sigmoid(tf.nn.bias_add(tf.matmul(all_inputc, W_bina), bias_bina))
                # update_gate = _binary_round(1.0 - tf.nn.relu(0.5 * (tf.nn.tanh(gate_bina + u1) + tf.nn.tanh(gate_bina - u1))))
                update_gate = _binary_round(gate_bina)

                # update_gate = tf.tile(update_gate, [1, self.num_units+1])

                new_c = c * tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)
                new_h = self._activation(new_c) * tf.sigmoid(o)  # always has value
                new_c = (1. - update_gate) * new_c
                # new_h = (1. - update_gate) * new_h_aws

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            # compas = tf.ones([256, self.num_units])
            # new_h_bgate = tf.concat([new_h, compas, update_gate], 1)
            new_h_bgate = tf.concat([new_h, update_gate], 1)

            return new_h_bgate, new_state


class CustomLSTMCell2(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        ''' t_max should be a float value corresponding to the longest possible
        time dependency in the input. '''
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        super(CustomLSTMCell2, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            x_size = x.get_shape().as_list()[1]
            batch_size = x.get_shape().as_list()[0]

            all_input = tf.concat([x, h], 1)
            all_inputc = tf.concat([x, c], 1)
            num_gates = 4

            W_xh = tf.get_variable('W_xh',
                                   [x_size + self.num_units, num_gates * self.num_units])
            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init_1long(self.t_max,
                                                               num_gates))
            W_bina = tf.get_variable('W_bina', [x_size + self.num_units, 1])
            bias_bina = tf.get_variable('bias_bina', [1],
                                        initializer=bias_initializer_reset(1))

            # u1 = tf.get_variable('u1', [1], initializer=init_u1(1))
            # D1 = tf.get_variable('D1', [1], initializer=init_D1(1))

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 4:  # divide c state
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)

                gate_bina = tf.sigmoid(tf.nn.bias_add(tf.matmul(all_inputc, W_bina), bias_bina))
                # update_gate = _binary_round(1.0 - tf.nn.relu(0.5 * (tf.nn.tanh(gate_bina + u1) + tf.nn.tanh(gate_bina - u1))))
                update_gate = _binary_round(gate_bina)

                # update_gate = tf.tile(update_gate, [1, self.num_units+1])

                new_c = c * tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)
                new_h = self._activation(new_c) * tf.sigmoid(o)  # always has value
                new_c = (1. - update_gate) * new_c
                # new_h = (1. - update_gate) * new_h_aws

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            # compas = tf.ones([256, self.num_units])
            # new_h_bgate = tf.concat([new_h, compas, update_gate], 1)
            new_h_bgate = tf.concat([new_h, update_gate], 1)

            return new_h_bgate, new_state


class ScdLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        ''' t_max should be a float value corresponding to the longest possible
        time dependency in the input. '''
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        super(ScdLSTMCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, update_gate_bottom, update_gate_top, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            x_size = x.get_shape().as_list()[1]

            all_input = tf.concat([x, h], 1)
            num_gates = 4

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

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 4:
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)
                update_gate_top = tf.expand_dims(update_gate_top, 1)
                update_gate_bottom = tf.expand_dims(update_gate_bottom, 1)
                new_c_tmp = c * tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)

                new_c = update_gate_bottom*new_c_tmp + (1 - update_gate_bottom)*c

                new_h = update_gate_bottom*self._activation(new_c) * tf.sigmoid(o)\
                        + (1 - update_gate_bottom)*h

                # new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j)
                # new_h = self._activation(new_c) * tf.sigmoid(o)
                # new_c = (1.-update_gate_top) * new_c
                # new_h = (1.-update_gate_top) * new_h_alw

                # compas = tf.ones([296, self.num_units])
                # new_h_com = tf.concat([new_h, compas], 1)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            return new_h, new_state


class ScdLSTMCell2(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        ''' t_max should be a float value corresponding to the longest possible
        time dependency in the input. '''
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        super(ScdLSTMCell2, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, update_gate_bottom, update_gate_top, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            x_size = x.get_shape().as_list()[1]

            all_input = tf.concat([x, h], 1)
            num_gates = 4

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

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 4:
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)
                update_gate_top = tf.expand_dims(update_gate_top, 1)
                update_gate_bottom = tf.expand_dims(update_gate_bottom, 1)
                new_c_tmp = c * tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)

                new_c = update_gate_bottom*new_c_tmp + (1 - update_gate_bottom)*c

                new_h = update_gate_bottom*self._activation(new_c) * tf.sigmoid(o)\
                        + (1 - update_gate_bottom)*h

                # new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(j)
                # new_h = self._activation(new_c) * tf.sigmoid(o)
                # new_c = (1.-update_gate_top) * new_c
                # new_h = (1.-update_gate_top) * new_h_alw

                # compas = tf.ones([296, self.num_units])
                # new_h_com = tf.concat([new_h, compas], 1)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            return new_h, new_state

class TrdLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=None, forget_only=False,
                 **kwargs):
        self.num_units = num_units
        self.t_max = t_max
        self.forget_only = forget_only
        super(TrdLSTMCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, input_nextt_gate, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            x_size = x.get_shape().as_list()[1]

            all_input = tf.concat([x, h], 1)
            num_gates = 4

            W_xh = tf.get_variable('W_xh',
                                   [x_size + self.num_units, num_gates * self.num_units])

            if self.t_max is None:
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=bias_initializer(num_gates))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [num_gates * self.num_units],
                                       initializer=chrono_init(self.t_max, num_gates))

            concat = tf.nn.bias_add(tf.matmul(all_input, W_xh), bias)

            if num_gates == 4:
                # i=input_gate, j=new_input, o=output_gate, f=forget_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=num_gates, axis=1)
                # i = tf.nn.bias_add(tf.matmul(c, W_ih), bias_ih)
                update_gate = tf.expand_dims(input_nextt_gate, 1)
                new_c = update_gate*(c*tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)) + (1 - update_gate)*c
                # new_c = c*tf.sigmoid(f)+tf.sigmoid(i)*self._activation(j)
                new_h = update_gate*self._activation(new_c) * tf.sigmoid(o)\
                        + (1 - update_gate)*h

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)

            return new_h, new_state


NTMControllerState = collections.namedtuple('NTMControllerState', ('controller_state', 'read_vector_list', 'w_list', 'M'))


class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 t_max=None, addressing_mode='content_and_location', shift_range=1, reuse=False, output_dim=None, clip_value=20,
                 init_mode='constant', keep_prob=0.9):
        self.controller_layers = controller_layers
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.t_max = t_max
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.clip_value = clip_value
        self.keep_prob = keep_prob

        # def single_cell(num_units):
        #     return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
        # self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(self.controller_units) for _ in range(self.controller_layers)])

        cell = TrdLSTMCell(self.controller_units, t_max=self.t_max, forget_only=True)
        single_cell = rnn_cell_impl_local_scd.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=self.keep_prob)
        self.controller = single_cell

        self.init_mode = init_mode

        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)

    def __call__(self, x, prev_state, update_gate, scope=None):
        prev_read_vector_list = prev_state.read_vector_list

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state.controller_state,
                                                                  update_gate)
            # controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)

        # NTM_output, NTMControllerState = self.prev_ntm(x, prev_state, controller_output, controller_state)
        NTM_output, NTMControllerState = tf.cond(update_gate[0] > 0.1,
                                                 lambda: self.conv_ntm(x, prev_state, controller_output, controller_state),
                                                 lambda: self.prev_ntm(x, prev_state, controller_output, controller_state))

        return NTM_output, NTMControllerState
        # return NTM_output, NTMControllerState(
        #     controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M)

    def prev_ntm(self, x, prev_state, controller_output, controller_state):

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            NTM_output = tf.contrib.layers.fully_connected(  #tf.concat([controller_output] + prev_state.read_vector_list, axis=1)
                controller_output, output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)
            NTM_output = tf.clip_by_value(NTM_output, -self.clip_value, self.clip_value)

        return NTM_output, NTMControllerState(
            controller_state=controller_state, read_vector_list=prev_state.read_vector_list, w_list=prev_state.w_list,
            M=prev_state.M)

    def transpose_for_scores(self, input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def reshape_to_matrix(self, input_tensor):
        """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
        ndims = input_tensor.shape.ndims
        if ndims < 2:
            raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                             (input_tensor.shape))
        if ndims == 2:
            return input_tensor

        width = input_tensor.shape[-1]
        output_tensor = tf.reshape(input_tensor, [-1, width])
        return output_tensor

    def dropout(self, input_tensor, dropout_prob):
        """Perform dropout.

        Args:
          input_tensor: float Tensor.
          dropout_prob: Python float. The probability of dropping out a value (NOT of
            *keeping* a dimension as in `tf.nn.dropout`).

        Returns:
          A version of `input_tensor` with dropout applied.
        """
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor

        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output

    def attention_layer(self, from_tensor,
                        attention_mask=None,
                        num_attention_heads=4,
                        size_per_head=None,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.3,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):

        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]

        # Scalar dimensions referenced here:
        #     B = batch size (number of sequences)
        #     F = `from_tensor` sequence length
        #     T = `to_tensor` sequence length
        #     N = `num_attention_heads`
        #     H = `size_per_head`

        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'query_layer' not in globals():
            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `key_layer` = [B*T, N*H]
        if 'key_layer' not in globals():
            key_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `value_layer` = [B*T, N*H]
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return from_tensor + context_layer  #[:, -1, :]


    def attention_layer2(self, from_tensor,
                        attention_mask=None,
                        num_attention_heads=4,
                        size_per_head=None,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.3,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):

        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]

        # Scalar dimensions referenced here:
        #     B = batch size (number of sequences)
        #     F = `from_tensor` sequence length
        #     T = `to_tensor` sequence length
        #     N = `num_attention_heads`
        #     H = `size_per_head`

        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'query_layer' not in globals():
            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query2",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `key_layer` = [B*T, N*H]
        if 'key_layer' not in globals():
            key_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key2",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `value_layer` = [B*T, N*H]
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value2",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return from_tensor + context_layer  #[:, -1, :]

    def attention_layer3(self, from_tensor,
                        attention_mask=None,
                        num_attention_heads=4,
                        size_per_head=None,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.3,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):

        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]

        # Scalar dimensions referenced here:
        #     B = batch size (number of sequences)
        #     F = `from_tensor` sequence length
        #     T = `to_tensor` sequence length
        #     N = `num_attention_heads`
        #     H = `size_per_head`

        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'query_layer' not in globals():
            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query3",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `key_layer` = [B*T, N*H]
        if 'key_layer' not in globals():
            key_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key3",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `value_layer` = [B*T, N*H]
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value3",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return from_tensor + context_layer  #[:, -1, :]

    def attention_layer4(self, from_tensor,
                        attention_mask=None,
                        num_attention_heads=4,
                        size_per_head=None,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.3,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):

        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]

        # Scalar dimensions referenced here:
        #     B = batch size (number of sequences)
        #     F = `from_tensor` sequence length
        #     T = `to_tensor` sequence length
        #     N = `num_attention_heads`
        #     H = `size_per_head`

        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'query_layer' not in globals():
            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query4",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `key_layer` = [B*T, N*H]
        if 'key_layer' not in globals():
            key_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key4",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `value_layer` = [B*T, N*H]
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value4",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return from_tensor + context_layer  #[:, -1, :]


    def attention_layer5(self, from_tensor,
                        attention_mask=None,
                        num_attention_heads=4,
                        size_per_head=None,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.3,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):

        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]

        # Scalar dimensions referenced here:
        #     B = batch size (number of sequences)
        #     F = `from_tensor` sequence length
        #     T = `to_tensor` sequence length
        #     N = `num_attention_heads`
        #     H = `size_per_head`

        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'query_layer' not in globals():
            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query5",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `key_layer` = [B*T, N*H]
        if 'key_layer' not in globals():
            key_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key5",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `value_layer` = [B*T, N*H]
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value5",
                # kernel_initializer=create_initializer(initializer_range)
            )

        # `query_layer` = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length,
                                           size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = self.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

        return from_tensor + context_layer  #[:, -1, :]


    def forward1(self, from_tensor,
                        num_attention_heads=None,
                        size_per_head=None,
                        value_act=tf.nn.relu,
                        ):
        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]
        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="f1",
            )
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = value_layer
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return from_tensor + context_layer
    def forward2(self, from_tensor,
                        num_attention_heads=None,
                        size_per_head=None,
                        value_act=tf.nn.relu,
                        ):
        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]
        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="f2",
            )
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = value_layer
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return from_tensor + context_layer
    def forward3(self, from_tensor,
                        num_attention_heads=None,
                        size_per_head=None,
                        value_act=tf.nn.relu,
                        ):
        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]
        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="f3",
            )
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = value_layer
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return from_tensor + context_layer
    def forward4(self, from_tensor,
                        num_attention_heads=None,
                        size_per_head=None,
                        value_act=tf.nn.relu,
                        ):
        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]
        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="f4",
            )
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = value_layer
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return from_tensor + context_layer
    def forward5(self, from_tensor,
                        num_attention_heads=None,
                        size_per_head=None,
                        value_act=tf.nn.relu,
                        ):
        batch_size = from_tensor.shape[0]
        from_seq_length = from_tensor.shape[1]
        to_seq_length = from_tensor.shape[1]
        from_tensor_2d = self.reshape_to_matrix(from_tensor)
        if 'value_layer' not in globals():
            value_layer = tf.layers.dense(
                from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="f5",
            )
        value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = value_layer
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return from_tensor + context_layer  #[:, -1, :]

    def conv_ntm(self, x, prev_state, controller_output, controller_state):
        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(
                controller_output, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)
            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        prev_w_list = prev_state.w_list
        prev_M = prev_state.M
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        # Reading (Sec 3.1)
        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)
        # Writing (Sec 3.2)
        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)
        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            # NTM_output = tf.contrib.layers.fully_connected(
            #     tf.concat([controller_output] + read_vector_list, axis=1), output_dim, activation_fn=None,
            #     weights_initializer=self.o2o_initializer)
            # NTM_output = tf.clip_by_value(NTM_output, -self.clip_value, self.clip_value)
            att = self.attention_layer(prev_M, num_attention_heads=4, size_per_head=self.output_dim//4)
            att = self.forward1(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.attention_layer2(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.forward2(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.attention_layer3(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.forward3(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.attention_layer4(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.forward4(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.attention_layer5(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            # att = self.forward5(att, num_attention_heads=4, size_per_head=self.output_dim // 4)
            #
            att_mean = tf.reduce_mean(att, axis=1, keepdims=False)

            NTM_output = tf.contrib.layers.fully_connected(
                att_mean, output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)  #tf.concat([controller_output] + [att[:, -1, :]], axis=1)

        self.step += 1
        return NTM_output, NTMControllerState(
            controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M)

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content
        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

        # Calculating w^c

        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5)

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope('init', reuse=self.reuse):
            read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim)), dim=0, N=batch_size)
                for i in range(self.read_head_num)]

            w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), dim=0, N=batch_size)
                for i in range(self.read_head_num + self.write_head_num)]

            controller_init_state = self.controller.zero_state(batch_size, dtype)

            if self.init_mode == 'learned':
                M = expand(tf.tanh(
                    tf.reshape(
                        learned_init(self.memory_size * self.memory_vector_dim),
                        [self.memory_size, self.memory_vector_dim])
                    ), dim=0, N=batch_size)
            elif self.init_mode == 'random':
                M = expand(
                    tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                    dim=0, N=batch_size)
            elif self.init_mode == 'constant':
                M = expand(
                    tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                        initializer=tf.constant_initializer(1e-6)),
                    dim=0, N=batch_size)

            return NTMControllerState(
                controller_state=controller_init_state,
                read_vector_list=read_vector_list,
                w_list=w_list,
                M=M)

    @property
    def state_size(self):
        return NTMControllerState(
            controller_state=self.controller.state_size,
            read_vector_list=[self.memory_vector_dim for _ in range(self.read_head_num)],
            w_list=[self.memory_size for _ in range(self.read_head_num + self.write_head_num)],
            M=tf.TensorShape([self.memory_size * self.memory_vector_dim]))

    @property
    def output_size(self):
        return self.output_dim


def chrono_init(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))

        # t_random = random_ops.random_uniform([1], minval=1.0,
        #                                      maxval=t_max, dtype=dtype, seed=42)
        # uni_vals2 = tf.log(random_ops.random_uniform([num_units], minval=1.0,
        #                                             maxval=t_random, dtype=dtype,
        #                                             seed=42))
        if num_gates == 2:
            bias_f = uni_vals2
            bias_co = -uni_vals2  # tf.zeros(num_units)
            return tf.concat([bias_f, bias_co], 0)

        elif num_gates == 3:
            bias_i = -uni_vals
            bias_j = tf.zeros(num_units)
            bias_f = uni_vals
            return tf.concat([bias_i, bias_j, bias_f], 0)

        elif num_gates == 4:
            bias_i = -uni_vals
            bias_j = tf.zeros(num_units)
            bias_f = uni_vals
            bias_o = tf.zeros(num_units)
            return tf.concat([bias_i, bias_j, bias_f, bias_o], 0)

        elif num_gates == 1:
            bias_j = 0.5*tf.ones(num_units)
            return bias_j

    return _initializer


def chrono_init_1long(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=1.0, dtype=dtype,
                                                    seed=42))
        if num_gates == 2:
            bias_f = uni_vals
            bias_j = tf.zeros(num_units)  # -uni_vals2
            return tf.concat([bias_f, bias_j], 0)
        elif num_gates == 4:
            bias_i = -uni_vals
            bias_j = tf.zeros(num_units)
            bias_f = uni_vals
            bias_o = tf.zeros(num_units)
            return tf.concat([bias_i, bias_j, bias_f, bias_o], 0)
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


def bias_initializer_reset(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        p = np.zeros(shape)
        num_units = int(shape[0] // num_gates)
        p[-num_units:] = 1*np.ones(num_units)
        return tf.constant(p, dtype)
    return _initializer