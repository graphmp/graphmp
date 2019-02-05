# -*- coding: utf-8 -*-
# !/usr/bin/env python


import numpy as np
import tensorflow as tf
import argparse
import math
import glob
import tensorflow.contrib.layers as layers
import os
import random
from tensorflow.contrib import rnn
from sklearn import metrics


def pairwise_distance(cate_pattern):
    """Compute pairwise distance of a point cloud.
    Args:
      cate_pattern: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size, num_points, num_dims = cate_pattern.get_shape().as_list()
    cate_pattern = tf.squeeze(cate_pattern)
    if og_batch_size == 1:
        cate_pattern = tf.expand_dims(cate_pattern, 0)

    cate_pattern_transpose = tf.transpose(cate_pattern, perm=[0, 2, 1])
    cate_pattern_inner = tf.matmul(cate_pattern, cate_pattern_transpose)
    cate_pattern_inner = -2 * cate_pattern_inner
    cate_pattern_square = tf.reduce_sum(tf.square(cate_pattern), axis=-1, keep_dims=True)
    cate_pattern_square_tranpose = tf.transpose(cate_pattern_square, perm=[0, 2, 1])
    return tf.reshape(cate_pattern_square + cate_pattern_inner + cate_pattern_square_tranpose, [-1, num_points, num_points])
# def pairwise_distance(cate_pattern):
#     """Compute pairwise distance of a point cloud.
#     Args:
#       cate_pattern: tensor (batch_size, num_points, num_dims)
#     Returns:
#       pairwise distance: (batch_size, num_points, num_points)
#     """
#     og_batch_size, num_points, num_dims = cate_pattern.get_shape().as_list()
#     cate_pattern = tf.squeeze(cate_pattern)
#     if og_batch_size == 1:
#         cate_pattern = tf.expand_dims(cate_pattern, 0)
#
#     cate_pattern_transpose = tf.transpose(cate_pattern, perm=[0, 2, 1])
#     cate_pattern_inner = tf.matmul(cate_pattern, cate_pattern_transpose)
#
#     return tf.reshape(cate_pattern_inner, [-1, num_points, num_points])


class pfn(object):

    def __init__(self, n_filters, filter_sizes,
                 top_down_channel , num_steps, attention_size, hidden_dim,
                 w=1, b=0, category = False):
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.attention_size = attention_size
        self.top_down_channel = top_down_channel
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.w = w
        self.b = b
        self.category = category
        self.neuron_per_layer = [32,32]


    def temporal_attention(self, inputs, str_name):
        print('inputs', inputs)
        inputs = tf.stack(inputs, axis = 0)
        print('inputs', inputs)
        inputs = tf.transpose(inputs, [1, 0, 2])

        with tf.variable_scope('attention' + str_name):
            inputs_shape = inputs.shape
            sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
            hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

            # Attention mechanism
            W_omega = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
            exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

            output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

            return output

    def RNN(self, x, num_steps, hidden_dim, scope):
        print(scope)
        # x.shape ->[batch_size, num_steps, n_category, n_channel]
        dim0, dim1, dim2, dim3 = x.get_shape().as_list()  # batch_size, input_dim
        x = tf.unstack(x, axis=1)
        x = x[-num_steps:]
        print('x_x', x)
        with tf.variable_scope(scope):
            # 1-layer LSTM with n_hidden units.
            conv_lstm_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims = 1,
                                                         input_shape= [dim2, dim3],
                                                         output_channels = hidden_dim,
                                                         kernel_shape = [1])
            # generate prediction
            outputs, states = rnn.static_rnn(conv_lstm_cell, x, dtype=tf.float32)
        return outputs[-1]

    def squeeze_excite_block(self, input, filters, name, ratio=1):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''

        init = input

        se = tf.reduce_mean(init, axis = 1)
        se = tf.layers.dense(se, filters // ratio, activation=tf.nn.relu,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se1'+name)
        se = tf.layers.dense(se, filters, activation=tf.nn.sigmoid,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se2'+name)

        x = tf.multiply(init, tf.expand_dims(se, axis = 1))
        return x, se

    def resnet_block(self, conv_output, layer_id, n_output):
        shortcut_connect_ = tf.layers.conv2d(conv_output,
                                       filters=n_output,
                                       kernel_size=(1,1),
                                       strides=(2,1),
                                       padding='same',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=self.regularizer,
                                       trainable=True,
                                       name='shortcut_layer_%s' % layer_id)

        conv_output1 = tf.layers.conv2d(conv_output,
                                       filters=n_output,
                                       kernel_size=(3, 1),
                                       strides=(1,1),
                                       padding='same',
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=self.regularizer,
                                       trainable=True,
                                       name='layer1_%s' % layer_id)

        conv_output2 = tf.layers.conv2d(conv_output1,
                                       filters=n_output,
                                       kernel_size=(3,1),
                                       strides=(1,1),
                                       padding='same',
                                       activation=None,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=self.regularizer,
                                       trainable=True,
                                       name='layer2_%s' % layer_id)

        conv_output2 = tf.layers.max_pooling2d(conv_output2,
                                              pool_size=(3,1),
                                              strides=(2,1),
                                              padding='same',
                                              name='layer_%s_maxpool' % layer_id)

        conv_output = tf.nn.relu(shortcut_connect_+conv_output2)
        return conv_output

    def squeeze_excite_block(self, input, filters, name, ratio=1):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''

        init = input

        se = tf.reduce_mean(init, axis = 1)
        se = tf.layers.dense(se, filters // ratio, activation=tf.nn.relu,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se1'+name)
        se = tf.layers.dense(se, filters, activation=tf.nn.sigmoid,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se2'+name)

        x = tf.multiply(init, tf.expand_dims(se, axis = 1))
        return x, se

    def bottle_up_path(self, sequence):
        dim0, dim1, dim2 = sequence.get_shape().as_list() #batch_size, input_dim
        tf.logging.info("dim0 %s, dim1 %s, dim2 %s" % (dim0, dim1, dim2))
        conv_output = tf.expand_dims(sequence, axis = 3)
        conv_outputs = {}; rnn_outputs = []
        conv_outputs[0] = conv_output

        rnn_output = self.RNN(conv_output, num_steps=self.num_steps, hidden_dim=self.hidden_dim,
                              scope='rnn_')
        tf.logging.info('conv_output', rnn_output.get_shape())
        rnn_output = tf.layers.dense(rnn_output,
                                     self.hidden_dim,
                                     activation=tf.nn.tanh,
                                     trainable=True,
                                     kernel_regularizer=self.regularizer,
                                     name='rnn_output')
        rnn_outputs.append(rnn_output)

        for layer_id, n_output in enumerate(self.n_filters):
            print('layer_%s' % layer_id)
            conv_output = self.resnet_block(conv_output, layer_id, n_output)
            conv_outputs[layer_id+1] = conv_output

            rnn_output = self.RNN(conv_output, num_steps = self.num_steps, hidden_dim = self.hidden_dim, scope = 'rnn_%s' % layer_id)
            tf.logging.info('conv_output', rnn_output.get_shape())
            rnn_output = tf.layers.dense(rnn_output,
                                         self.hidden_dim,
                                          activation=tf.nn.tanh,
                                          trainable=True,
                                          kernel_regularizer=self.regularizer,
                                          name='rnn_output%s' %layer_id)
            rnn_outputs.append(rnn_output)
        return rnn_output

    def conv2d_transpose(self, input, kernel_size, depth_output, stride, name):
        dim0, dim1, dim2, dim3 = input.get_shape().as_list()  # batch_size, n_time, n_category, depth
        print('dim0, dim1, dim2, dim3', dim0, dim1, dim2, dim3)
        with tf.variable_scope(name):
            k = tf.get_variable('kernel', shape = [kernel_size, 1, depth_output, dim3],
                                dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())  # note k.shape = [cols, depth_output, depth_in]
            output_shape = tf.stack([tf.shape(input)[0], dim1*2, dim2, depth_output])
            output = tf.nn.conv2d_transpose(value=input,
                                              filter=k,
                                              output_shape=output_shape,
                                              strides=(1,stride,1,1),
                                              padding='SAME')

            output = tf.nn.relu(output)
        return output

    def top_down_path(self, conv_outputs):
        rnn_outputs = []
        print('conv_outputs', conv_outputs)

        for reverse_layer_id in range(len(conv_outputs)-1, 0, -1):

            'lateral connection'

            lateral_layer = conv_outputs[reverse_layer_id-1]
            lateral_layer_connect = tf.layers.conv2d(lateral_layer,
                                                   filters=self.top_down_channel,
                                                   kernel_size=(1,1),
                                                   strides=(1,1),
                                                   padding='same',
                                                   activation=None,
                                                   use_bias=True,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.contrib.layers.xavier_initializer(),
                                                   kernel_regularizer=self.regularizer,
                                                   trainable=True,
                                                   name='lateral_layer_%s' % reverse_layer_id)
            if reverse_layer_id>0:
                conv_output = self.conv2d_transpose(conv_outputs[reverse_layer_id], kernel_size = self.filter_sizes[reverse_layer_id-1],
                                                    depth_output = self.top_down_channel, stride = 2, name = 'topdown_%s_bn' % reverse_layer_id)
            else:
                conv_output = self.conv2d_transpose(conv_outputs[reverse_layer_id],
                                                    kernel_size=1,
                                                    depth_output=self.top_down_channel, stride=2,
                                                    name='topdown_%s_bn' % reverse_layer_id)

            conv_output = lateral_layer_connect + conv_output

            # 'mitigate the overlapping effect caused by upsampling conv1d'
            # conv_output = tf.layers.conv1d(conv_output,
            #                                filters=self.top_down_channel,
            #                                kernel_size=3,
            #                                strides=1,
            #                                padding='same',
            #                                activation=tf.nn.relu,
            #                                use_bias=True,
            #                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                bias_initializer=tf.contrib.layers.xavier_initializer(),
            #                                kernel_regularizer=self.regularizer,
            #                                trainable=True,
            #                                name='output_layer%s' % reverse_layer_id)



            'Encode the temporal pattern from different resolution'
            rnn_output = self.RNN(conv_output, num_steps = self.num_steps, hidden_dim = self.hidden_dim, scope = 'rnn_%s' % reverse_layer_id)
            tf.logging.info('conv_output', rnn_output.get_shape())
            rnn_output = tf.layers.dense(rnn_output,
                                         self.hidden_dim,
                                          activation=tf.nn.tanh,
                                          trainable=True,
                                          kernel_regularizer=self.regularizer,
                                          name='rnn_output%s' %reverse_layer_id)
            rnn_outputs.append(rnn_output)
        return rnn_outputs



    def hie_fusion(self, rnn_outputs):
        intermediate_layer = rnn_outputs[0]
        for idx, rnn_output in enumerate(rnn_outputs[1:]):
            intermediate_layer = tf.layers.dense(tf.concat([rnn_output, intermediate_layer], axis = 1),
                                         self.hidden_dim,
                                          activation=tf.nn.relu,
                                          trainable=True,
                                          kernel_regularizer=self.regularizer,
                                          name='hie_output%s' %idx)
        return intermediate_layer

    def gcn_op(self, train_features):
        l_ = train_features
        for idx, n_neuron in enumerate(self.neuron_per_layer):
            train_adj = -pairwise_distance(l_)
            l_ = tf.layers.dense(tf.matmul(train_adj, l_),
                                 n_neuron,
                                 activation=tf.nn.relu,
                                 trainable=True,
                                 kernel_regularizer=self.regularizer,
                                 name='pred_v'+str(idx))
        return l_


    def pfn_cnn(self, category_dim, sequence):
        mult_reolution_embeds = self.bottle_up_path(sequence)


        # 'encode mult_reolution_embeds and then output predicted value'
        # 'TODO: multiple temporal pattern fusion --attention, other way?'
        concat_reolution_embeds = tf.concat(mult_reolution_embeds, axis = 2)#output: batch_size,  category, channel
        # resign weight
        concat_reolution_embeds, weights = self.squeeze_excite_block(concat_reolution_embeds, concat_reolution_embeds.get_shape().as_list()[2], 'se_%s' % 0,
                                                              ratio=1)  # output: batch_size,  category, channel


        'normalize adj mtx'
        # deg_matrix = tf.matrix_diag(tf.pow(tf.reduce_sum(adj_matrix, axis = -1), -0.5))
        # adj_matrix = tf.matmul(tf.matmul(deg_matrix, adj_matrix), deg_matrix)
        # print('adj', adj_matrix)

        propagate_resolution = self.gcn_op(concat_reolution_embeds)


        pred_v = tf.squeeze(tf.layers.dense(propagate_resolution,
                                 1,
                                 activation=None,
                                 trainable=True,
                                 kernel_regularizer=self.regularizer,
                                 name='pred_v'), axis = -1)

        return pred_v, concat_reolution_embeds

    def evaluate(self, preds, ys):
        def confusion_mtx(pred, y):
            TP = tf.count_nonzero(pred * y, dtype=tf.float32)
            TN = tf.count_nonzero((pred - 1) * (y - 1), dtype=tf.float32)
            FP = tf.count_nonzero(pred * (y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((pred - 1) * y, dtype=tf.float32)
            return TP, TN, FP, FN
        sep_TP = []
        sep_TN = []
        sep_FP = []
        sep_FN = []
        for pred, y in zip(preds, ys):
            TP, TN, FP, FN = confusion_mtx(pred, y)
            sep_TP.append(TP)
            sep_TN.append(TN)
            sep_FP.append(FP)
            sep_FN.append(FN)
        preds = tf.reshape(tf.stack(preds), [-1])
        ys = tf.reshape(tf.stack(ys), [-1])
        allTP, allTN, allFP, allFN = confusion_mtx(preds, ys)
        return sep_TP, sep_TN, sep_FP, sep_FN, allTP, allTN, allFP, allFN

    def model(self, input_dim, category_dim, lr= 1e-3, decay=False):
        def my_tf_round(x, decimals=0):
            multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
            return tf.round(x * multiplier) / multiplier

        train_xs = tf.placeholder(tf.float32, shape=[None, input_dim, category_dim], name="train_xs")
        train_ys = tf.placeholder(tf.float32, shape=[None, category_dim], name="train_ys")
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        fusion_mtx = 0

        global_step = tf.Variable(0, name='global_step', trainable=False)

        pred_y, weights = self.pfn_cnn(category_dim, train_xs)

        if self.category:
            label = tf.math.greater(train_ys, -0.999)
            loss = tf.losses.sigmoid_cross_entropy(
                    label,
                    pred_y,
                    weights=5.0)
            scale_back_train_ys = tf.unstack(tf.cast(label, tf.int32), axis = 1)
            scale_back_pred_binary_y = tf.unstack(tf.cast(tf.math.greater(tf.sigmoid(pred_y), 0.5), tf.int32), axis=1)

            sep_TP, sep_TN, sep_FP, sep_FN, allTP, allTN, allFP, allFN = self.evaluate(scale_back_pred_binary_y,
                                                                                       scale_back_train_ys)

            fusion_mtx = (sep_TP, sep_TN, sep_FP, sep_FN, allTP, allTN, allFP, allFN)
            scale_back_pred_y = tf.unstack(my_tf_round(tf.sigmoid(pred_y), 2), axis=1)
        else:
            loss = tf.reduce_mean(tf.squared_difference(train_ys, pred_y))
            scale_back_train_ys = tf.divide(tf.add(train_ys, -self.b), self.w)
            scale_back_pred_y = tf.divide(tf.add(pred_y, -self.b), self.w)
        l2_loss = tf.losses.get_regularization_loss()
        loss += 0.0*l2_loss



        saver = tf.train.Saver(max_to_keep=1)

        #GradientDescentOptimizer, AdamOptimizer
        if decay:
            opt = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(lr, global_step,
                                                                                  decay_steps=1000, decay_rate=0.99,
                                                                                  staircase=True))
        else:
            opt = tf.train.AdamOptimizer(learning_rate=lr)

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        # train_op = opt.minimize(loss, global_step=global_step)
        gvs = opt.compute_gradients(loss)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        train_op = opt.apply_gradients(capped_gvs, global_step=global_step)

        init = tf.global_variables_initializer()
        return train_xs, train_ys, scale_back_train_ys, scale_back_pred_y, weights, saver, \
               loss, self.train_phase, train_op, init, global_step, fusion_mtx


def random_sample(batch_size, input_dim, category_dim, n_epoch):

    def load():
        for idx, _ in enumerate(range(100)):
            xs = np.random.rand(batch_size, input_dim, category_dim)
            ys = np.random.rand(batch_size, category_dim)
            yield (xs, ys)

    for epoch in range(n_epoch):
        yield load()


class pfnIO:
    def __init__(self, n_filters, filter_sizes,top_down_channel,
                   num_steps, attention_size, hidden_dim, w = 1, b = 0, input_dim=256, category_dim = 20, batch_size=64,
                 keep_rate=0.6,
                 learning_rate=1e-3,
                 category = False):  # n is number of users
        # map_id_idx = map_dict([], args.dict)
        # self.map_idx_id = {v: k for k, v in map_id_idx.items()}
        # num_nodes = max(self.map_idx_id.values())
        # print(num_nodes)
        self.input_dim = input_dim
        self.category_dim = category_dim
        self.batch_size = batch_size
        self.keep_rate = keep_rate
        self.category_flag = category

        pfn_ = pfn(n_filters, filter_sizes,top_down_channel,
                   num_steps, attention_size, hidden_dim,
                   w, b, category)
        self.train_xs, self.train_ys, self.scale_back_train_ys, self.scale_back_pred_y, self.weight, self.saver, \
        self.loss, self.train_phase, self.train_op, self.init, self.global_step, self.fusion_mtx = pfn_.model(input_dim, category_dim, lr=learning_rate, decay=False)

        self.user_batch = []
        self.article_batch = []
        self.click_batch = []

    def create(self, pretrain_flag=0, save_file=''):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.log_device_placement = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        if pretrain_flag == 0:
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, save_file + '.ckpt')
        print('create')

    def save_weight(self, save_file):
        try:
            self.saver.save(self.sess, save_file+'.ckpt')
        except:
            print('need to makedir')
            os.makedirs(save_file, exist_ok=True)
            self.saver.save(self.sess, save_file+'.ckpt')
        print("Save model to file as pretrain.")

    def ExpUpdateParameters(self, n_epoch=1):
        average_loss = 0
        for epoch in random_sample(self.batch_size, self.input_dim, self.category_dim, n_epoch):
            for (xs, ys) in epoch:
                (step, loss) = self.feedbatch(xs, ys)
                average_loss += loss
                if step % 50 == 0:
                    average_loss /= 50;
                    print('step %s, avg_loss %.3f, curr_loss %.3f' %(step, average_loss, loss))
                    average_loss = 0;


    def run_eval(self, rs, sequence_vs, sequence_open, ys):
        xs_batches = 1 + len(ys) // 10000;
        my_preds = []
        my_trues = []
        for idx in range(xs_batches):
            idx_begin = idx * 10000
            idx_end = min((idx + 1) * 10000, len(ys))

            my_true, my_pred =  self.get_prediction(sequence_vs[idx_begin:idx_end],ys[idx_begin:idx_end])
            my_preds = my_preds + list(my_pred)
            my_trues = my_trues + list(my_true)
        my_rmse = np.sqrt(metrics.mean_squared_error(np.array(my_trues), np.array(my_preds)))
        rmspe = np.sqrt(np.mean([((p - t) / t) ** 2 for p, t in zip(my_preds, my_trues) if t > 10]))
        my_mae = metrics.mean_absolute_error(np.array(my_trues), np.array(my_preds))
        my_cr = 1 - my_mae / np.mean(my_trues)
        print('fpn-pred', np.round(np.array(my_preds[:10]),0))
        print('fpn-gt',  np.round(np.array(my_trues[:10]),0))
        return ([my_rmse, rmspe, my_mae, my_cr])

    def eval_rslt(self, TP, TN, FP, FN):
        prec = TP / (TP + FP + 0.1)
        recall = TP / (TP + FN + 0.1)
        f1 = 2 * TP / (2 * TP + FN + FP + 0.1)
        acc = (TP + TN) / (TP + FN + FP + TN + 0.1)
        return (round(float(prec), 4), round(float(recall), 4), round(float(f1), 4), round(float(acc), 4))

    def run_binary_eval(self, rs, sequence_vs, ys, category):
        xs_batches = 1 + len(ys) // 5000;
        my_preds = {c:[] for c in category}; my_trues = {c:[] for c in category}; my_fusions = []
        for idx in range(xs_batches):
            idx_begin = idx * 5000
            idx_end = min((idx + 1) * 5000, len(rs))
            my_true, my_pred, my_fusion =  self.get_prediction(sequence_vs[idx_begin:idx_end],ys[idx_begin:idx_end], category = self.category_flag)
            for c, t, p in zip(category, my_true, my_pred):
                my_preds[c] = my_preds[c] + list(p)
                my_trues[c] = my_trues[c] + list(t)
                my_fusions.append(my_fusion)

        self.sep_TP = [0]*self.category_dim; self.sep_TN = [0]*self.category_dim; self.sep_FP = [0]*self.category_dim; self.sep_FN = [0]*self.category_dim;
        allTP = 0; allTN = 0; allFP = 0; allFN = 0;
        for _ in my_fusions:
            self.sep_TP = [org_c + up_c for org_c, up_c in zip(self.sep_TP, _[0])]
            self.sep_TN = [org_c + up_c for org_c, up_c in zip(self.sep_TN, _[1])]
            self.sep_FP = [org_c + up_c for org_c, up_c in zip(self.sep_FP, _[2])]
            self.sep_FN = [org_c + up_c for org_c, up_c in zip(self.sep_FN, _[3])]
            allTP += _[4]; allTN += _[5]; allFP += _[6]; allFN += _[7];

        self.my_trues = my_trues
        self.my_preds = my_preds

        my_true = np.reshape(np.array(list(my_trues.values())), [-1])
        my_pred = np.reshape(np.array(list(my_preds.values())), [-1])
        (prec, recall, f1, acc) = self.eval_rslt(allTP, allTN, allFP, allFN)

        fpr, tpr, thresholds = metrics.roc_curve(my_true, my_pred, pos_label=1)
        my_auc = metrics.auc(fpr, tpr)
        my_ap = metrics.average_precision_score(my_true, my_pred)
        print('======>', 'prec', prec, 'recall', recall, 'f1', f1, 'acc', round(float(my_auc), 4), 'ap',
              round(float(my_ap), 4))
        return [prec, recall, f1, acc, my_auc, my_ap]

    def run_sep_eval(self, category):
        for idx, c in enumerate(category):
            print(self.sep_TP[idx], self.sep_TN[idx], self.sep_FP[idx], self.sep_FN[idx])
            prec, recall, f1, acc= self.eval_rslt(self.sep_TP[idx], self.sep_TN[idx], self.sep_FP[idx], self.sep_FN[idx])
            fpr, tpr, thresholds = metrics.roc_curve(np.array(self.my_trues[c]), np.array(self.my_preds[c]), pos_label=1)
            my_auc = metrics.auc(fpr, tpr)
            my_ap = metrics.average_precision_score(np.array(self.my_trues[c]), np.array(self.my_preds[c]))
            print('%s: my_rmse %.4f , my_recall %.4f , my_f1 %.4f , my_acc %.4f , my_auc %.4f , my_ap %.4f' % (
                c, prec, recall, f1, acc, my_auc, my_ap))


    def feedbatch(self, xs, ys):
        feed_dict = {self.train_xs: xs,
                     self.train_ys: ys,
                     self.train_phase: 1}
        _, step, loss = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict)
        return (step, loss)

    def get_prediction(self, xs, ys, category = False):
        feed_dict = {self.train_xs: xs,
                     self.train_ys: ys,
                     self.train_phase: 0}
        if category:
            gt_, pred_, fusion_mtx = self.sess.run([self.scale_back_train_ys, self.scale_back_pred_y, self.fusion_mtx], feed_dict)
            return gt_, pred_, fusion_mtx
        else:
            gt_, pred_ = self.sess.run([self.scale_back_train_ys, self.scale_back_pred_y], feed_dict)
            return np.reshape(gt_, [-1]), np.reshape(pred_, [-1])

    def get_weight(self, xs, ys):
        feed_dict = {self.train_xs: xs,
                     self.train_ys: ys,
                     self.train_phase: 0}
        gt_, pred_, weight_ = self.sess.run([self.scale_back_train_ys, self.scale_back_pred_y, self.weight], feed_dict)
        return np.reshape(gt_, [-1]), np.reshape(pred_, [-1]), weight_




if __name__ == '__main__':
    #input: n_sequence, category
    nn = pfnIO(n_filters = [32, 64, 128, 256], filter_sizes = [3, 8, 5, 1],
               top_down_channel = 64, num_steps = 5, attention_size = 64,
               w=1, b=0,
               hidden_dim = 64,input_dim=256, batch_size=64,
               keep_rate=0.6,
               learning_rate=1e-3)
    nn.create()
    nn.ExpUpdateParameters(n_epoch=10)