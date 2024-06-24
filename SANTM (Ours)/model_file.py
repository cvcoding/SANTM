import tensorflow as tf
from aux_code.rnn_cells import CustomLSTMCell, CustomLSTMCell2, NTMCell, ScdLSTMCell, ScdLSTMCell2
from aux_code.tf_ops import linear
import rnn_local, rnn_local_new, rnn_local_doublenew
import rnn_cell_impl_local, rnn_cell_impl_local2, rnn_cell_impl_doublelocal
from tensorflow.python.framework import ops
import numpy as np
from tensorflow.python.ops import random_ops


def rnn(x, h_dim, y_dim, keep_prob, mask_noround, u, sequence_lengths,
        training, output_format, cell_type='janet',
        t_max=None, batch_size=0):

    def _binary_round(x):
        g = tf.get_default_graph()
        with ops.name_scope("BinaryRound") as name:
            with g.gradient_override_map({"Round": "Identity"}):
                return tf.round(x, name=name)

    def init_D(t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            bias_j = 1 * tf.ones([t_max])  # 0.5
            return bias_j
        return _initializer

    def fst_cell(dim, output_projection=None):
        cell = CustomLSTMCell(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_local.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    def fst_cell2(dim, output_projection=None):
        cell = CustomLSTMCell2(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_local.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    def scd_cell(dim, output_projection=None):
        cell = ScdLSTMCell(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_doublelocal.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    def scd_cell2(dim, output_projection=None):
        cell = ScdLSTMCell2(dim, t_max=t_max, forget_only=True)
        drop_cell = rnn_cell_impl_doublelocal.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell


    def trd_cell(dim, output_projection=None):
        cell = NTMCell(1, dim, 32, dim,  # 1, 32, 64
                       1, 1, t_max=t_max, addressing_mode='content_and_location',
                       shift_range=1, reuse=False, output_dim=dim,
                       clip_value=20, init_mode='learned', keep_prob=keep_prob)
        # __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim,
        #           read_head_num, write_head_num, t_max=None, addressing_mode='content_and_location',
        #           shift_range=1, reuse=False, output_dim=None,
        #          clip_value=20, init_mode='constant', keep_prob=0.9):
        drop_cell = rnn_cell_impl_local2.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    fstcell = fst_cell(h_dim[0])
    # fstcell2 = fst_cell2(h_dim[0])
    scdcell = scd_cell(h_dim[1])
    # scdcell2 = scd_cell2(h_dim[1])
    trdcell = trd_cell(h_dim[1])

    if output_format == 'last':

        D = tf.get_variable('D', [t_max], initializer=init_D(t_max))
        D_diag = tf.matrix_diag(D)

        mask2 = _binary_round(mask_noround)

        mask2_weighted = tf.matmul(D_diag, mask2)
        mask2 = tf.expand_dims(mask2, 0)

        mask2 = tf.tile(mask2, [batch_size, 1, h_dim[1]])
        mask2dim_out = mask2[:, :, 0]
        mask2asinput = tf.expand_dims(mask2[:, :, 0], 2)

        mask2_weighted = tf.expand_dims(mask2_weighted, 0)
        mask2_weighted = tf.tile(mask2_weighted, [batch_size, 1, 1])

        ###processing data
        out, _ = rnn_local.dynamic_rnn(
            fstcell, x, sequence_length=sequence_lengths,
            dtype=tf.float32)

        # out2, _ = rnn_local_new.dynamic_rnn(
        #     fstcell2, out, sequence_length=sequence_lengths,
        #     dtype=tf.float32)
        # maskfrom1lay = tf.expand_dims(out[:, :, -1], 2)
        # out = tf.concat([out2, maskfrom1lay], 2)   # for compasate the place of one mask

        # out, _ = rnn_local.dynamic_rnn(
        #     fstcell2, out, sequence_length=sequence_lengths,  #[:, :, 0: h_dim[1]-1]
        #     dtype=tf.float32)

        pointer_1st = tf.expand_dims(out[:, :, -1], 2)

        new_x = tf.concat([out, mask2asinput], 2)  # mask1: bottom; mask2: top
        out, _ = rnn_local_doublenew.dynamic_rnn(
            scdcell, new_x, sequence_length=sequence_lengths,
            dtype=tf.float32)
        out = out * mask2_weighted


        # add one more inter layer (scd layer)
        # new_x = tf.concat([out, pointer_1st], 2)
        # new_x = tf.concat([new_x, mask2asinput], 2)
        # out, _ = rnn_local_doublenew.dynamic_rnn(scdcell2, new_x, sequence_length=sequence_lengths, dtype=tf.float32)



        new_out = tf.concat([out, mask2asinput], 2)

        outputs, _ = rnn_local_new.dynamic_rnn(
            trdcell, new_out, sequence_length=sequence_lengths,
            dtype=tf.float32)
        out = outputs[:, -1, :]
        # if cell_type == 'lstm' or cell_type == 'Reslstm':
        #     out = out[1]
        proj_out = linear(out, y_dim, scope='output_mapping')

    elif output_format == 'all':
        out, _ = tf.nn.dynamic_rnn(
            fstcell, x, sequence_length=sequence_lengths,
            dtype=tf.float32)
        flat_out = tf.reshape(out, (-1, out.get_shape()[-1]))
        proj_out = linear(flat_out, y_dim, scope='output_mapping')
        proj_out = tf.reshape(proj_out, (tf.shape(out)[0], tf.shape(out)[1], y_dim))

    return proj_out, mask2dim_out    #


class RNN_Model(object):

    def __init__(self, n_features, n_classes, h_dim, max_sequence_length,
                 is_test=False, max_gradient_norm=None, opt_method='adam',
                 learning_rate=0.001, weight_decay=0,
                 cell_type='lstm', chrono=False, mse=False, batch_size=0,
                 ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.h_dim = h_dim
        self.max_sequence_length = max_sequence_length
        self.opt_method = opt_method
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.is_test = is_test
        self.weight_decay = weight_decay
        self.cell_type = cell_type
        self.chrono = chrono
        self.mse = mse
        self.batch_size = batch_size

    def init_mask(self, t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            t_random = random_ops.random_uniform([t_max, 1], minval=0, maxval=1.0, dtype=dtype, seed=42)
            bias_j = t_random
            # bias_j0 = 1.0*tf.ones([t_max-1, 1])   # 0.01
            # bias_j1 = 0.01*tf.ones([1, 1])
            # bias_j = tf.concat([bias_j0, bias_j1], 0)

            return bias_j
        return _initializer

    def init_u(self, t_max):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            bias_j = 0.1*tf.ones([t_max, 1])
            return bias_j
        return _initializer

    def build_inputs(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.x = tf.placeholder(tf.float32, [None, None, self.n_features],
                                name='x')
        if self.output_seq:
            self.y = tf.placeholder(tf.float32, [None, None, self.n_classes],
                                    name='y')
        else:
            self.y = tf.placeholder(tf.float32, [None, self.n_classes],
                                    name='y')
        self.seq_lens = tf.placeholder(tf.int32, [None],
                                       name="sequence_lengths")
        self.training = tf.placeholder(tf.bool)

        # self.mask_noround = tf.placeholder(tf.float32, [None, self.n_features],
        #                             name='mask_noround')
        self.mask_noround = tf.get_variable('mask_noround', [self.max_sequence_length, 1], initializer=self.init_mask(self.max_sequence_length))  #
        self.u = tf.get_variable('u', [self.max_sequence_length, 1], initializer=self.init_u(self.max_sequence_length))

        self.mask_noround = 1 - tf.nn.relu(0.5 * (tf.nn.tanh(self.mask_noround + self.u) + tf.nn.tanh(self.mask_noround - self.u)))

        # self.mask_noround = tf.sigmoid(self.mask_noround)

    def build_loss(self, outputs, mask):
        if self.mse:
            mean_squared_error = tf.losses.mean_squared_error(
                labels=self.y, predictions=outputs)
            self.loss_nowd = tf.reduce_mean(mean_squared_error)
            tf.summary.scalar('mean_squared_error',
                              tf.reduce_mean(mean_squared_error))
        else:
            if self.output_seq:
                flat_out = tf.reshape(outputs, (-1, tf.shape(outputs)[-1]))
                flat_y = tf.reshape(self.y, (-1, tf.shape(self.y)[-1]))
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=flat_y, logits=flat_out)

            else:
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y, logits=outputs)

            tf.summary.scalar('cross_entropy',
                              tf.reduce_mean(sample_cross_entropy))
            self.loss_nowd = tf.reduce_mean(sample_cross_entropy)

        weight_decay = self.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in
                                                   tf.trainable_variables()
                                                   if 'bias' not in v.name])
        tf.summary.scalar('weight_decay', weight_decay)

        # mask_size = mask.get_shape().as_list()[1]
        # mask_half = mask[:, round(mask_size / 2):-1]

        # mask_norm = 1e-6*tf.norm(mask, 1)  #tf.nn.l2_loss(mask)
        self.loss = self.loss_nowd + weight_decay #+ mask_norm


    def build_optimizer(self):
        if self.opt_method == 'adam':
            print('Optimizing with Adam')
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opt_method == 'rms':
            print('Optimizing with RMSProp')
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.opt_method == 'momentum':
            print('Optimizing with Nesterov momentum SGD')
            opt = tf.train.MomentumOptimizer(self.learning_rate,
                                             momentum=0.9,
                                             use_nesterov=True)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        tf.summary.scalar('gradients_norm', norm)

        self.train_opt = opt.apply_gradients(zip(clipped_gradients, params))
        # #  sparse project of the mask.

    def build(self, output_format='last'):
        print("Building model ...")
        self.output_seq = False
        if output_format == 'all':
            self.output_seq = True
        self.build_inputs()

        t_max = None
        if self.chrono:
            t_max = self.max_sequence_length

        outputs, mask = rnn(self.x, self.h_dim, self.n_classes, self.keep_prob, self.mask_noround, self.u,
                      sequence_lengths=self.seq_lens,
                      training=self.training, output_format=output_format,
                      cell_type=self.cell_type, t_max=t_max, batch_size=self.batch_size, )
        self.mask = mask

        self.build_loss(outputs, mask)

        self.output_probs = tf.nn.softmax(outputs)
        if self.output_seq:
            self.output_probs = tf.reshape(
                self.output_probs, (-1, tf.shape(self.output_probs)[-1]))

        if not self.is_test:
            print("Adding training operations")
            self.build_optimizer()

        # # use another sparse function relu[0.5[tanh(x+u)+tanh(x-u)]]
        # mask_sparse = 1 - tf.nn.relu(0.5 * (tf.nn.tanh(self.mask_noround + self.u) + tf.nn.tanh(self.mask_noround - self.u)))
        # mask_sparse = tf.transpose(mask_sparse)
        # top_k = 700
        # a_top, a_top_idx = tf.nn.top_k(mask_sparse, top_k, sorted=False)
        # kth = tf.reduce_min(a_top, axis=1, keepdims=True)
        # top2 = tf.greater_equal(mask_sparse, kth)
        # mk = tf.cast(top2, dtype=tf.float32)
        # self.mask_noround = tf.transpose(tf.multiply(mask_sparse, mk))
        # self.mask_noround = tf.to_float(self.mask_noround)

        self.summary_op = tf.summary.merge_all()
