#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@


import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.training import training
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as att_w

__all__ = ['_conv2d', '_conv3d', '_relu', '_max_pool_2d', '_max_pool_3d',
           '_channel_wise_max_pool', '_batch_norm', '_fc', '_dropout',
           '_mutli_layer_rnn', '_attention_decoder_wrapper', '_eltwise_sum',
           '_rns', '_eltwise_sum_conv2d']

def _conv2d(_input, out_channels, kh=5, kw=5, sh=2, sw=2, stddev=0.01, padding='SAME',
           name="conv2d", dtype=dtypes.float32, bias_add=True):
    """
    A wrapped conv2d operation
    :param _input: tensor, shape = [batch_size, height, width, channels]
    :param out_channels: scalar, convolution output channels
    :param kh: scalar, filter height
    :param kw: scalar, filter width
    :param sh: scalar, stride y
    :param sw: scalar, stride x
    :param stddev: scalar, standard deviation used for params' initialization
    :param padding: string, 'VALID' or 'SAME'
    :param bias_add: bool, whether to add bias to convolution result
    :return:
        tensor, convolution result which has the same shape length as _input
    """
    with variable_scope.variable_scope(name) as scope:
        weights = variable_scope.get_variable('w', [kh, kw, _input.shape[-1].value, out_channels], dtype=dtype,
                        initializer=init_ops.truncated_normal_initializer(stddev=stddev, dtype=dtype, seed=20170705))

        conv = nn_ops.conv2d(_input, weights, strides=[1, sh, sw, 1], padding=padding)

        if bias_add:
            biases = variable_scope.get_variable('biases', [out_channels],
                                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype),
                                                 dtype=dtype)
            return nn_ops.bias_add(conv, biases)
        else:
            return conv

def _conv3d(_input, out_channels, kd=3, kh=3, kw=3, sd=1, sh=1, sw=1, stddev=0.01, padding='SAME',
           name="conv2d", dtype=dtypes.float32, bias_add=True):
    """
    A wrapped conv3d operation
    :param _input: tensor, shape = [batch_size, depth, height, width, channels]
    :param out_channels: scalar, convolution output channels
    :param kd: scalar, filter depth
    :param kh: scalar, filter height
    :param kw: scalar, filter width
    :param sd: scalar, stride depth
    :param sh: scalar, stride y
    :param sw: scalar, stride x
    :param stddev: scalar, standard deviation used for params' initialization
    :param padding: string, 'VALID' or 'SAME'
    :param bias_add: bool, whether to add bias to convolution result
    :return:
        tensor, convolution result which has the same shape length as _input
    """
    with variable_scope.variable_scope(name):
        weights = variable_scope.get_variable('w', [kd, kh, kw, _input.shape[-1].value, out_channels], dtype=dtype,
                                              initializer=init_ops.truncated_normal_initializer(stddev=stddev,
                                                                                                dtype=dtype,
                                                                                                seed=20170705))
        conv = nn_ops.conv3d(_input, weights, strides=[1, sd, sh, sw, 1], padding=padding)
        if bias_add:
            biases = variable_scope.get_variable('biases', [out_channels],
                                                 initializer=init_ops.constant_initializer(0.0, dtype=dtype),
                                                 dtype=dtype)
            return nn_ops.bias_add(conv, biases)
        else:
            return conv

def _relu(_input, name="relu"):
    return nn_ops.relu(_input, name=name)

def _max_pool_2d(_input, kh=2, kw=2, sh=2, sw=2, name="max_pool_2d", padding='SAME'):
    return nn_ops.max_pool(_input, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding=padding, name=name)

def _max_pool_3d(_input, kd=2, kh=2, kw=2, sd=2, sh=2, sw=2, name="max_pool_3d", padding='SAME'):
  return nn_ops.max_pool3d(_input, ksize=[1, kd, kh, kw, 1], strides=[1, sd, sh, sw, 1], padding=padding, name=name)

def _channel_wise_max_pool(_input, keep_dims=False):
    """
    channel-wise max pooling operation
    :param _input: tensor with ndim = 3 or 4 or 5, and the last dimension is channel
    :return:
        a tensor after  channel-wise max pooling operation which has shape = input.shape[:-1]

    **warning**:
        here, reduction_indices is the old name of axis,
        in order to work in legacy version of tensorflow
    """
    return tf.reduce_max(_input, reduction_indices=[-1], keep_dims=keep_dims)


def _batch_norm(_input, trnFlag, eps=1e-3, name="batch_norm", ema_decay=0.5, dtype=dtypes.float32):
    """
    A wrapped BN operation used for 2D or 3D convolution as described in:

        https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
        https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow?answertab=votes#tab-top

    :param _input: tensor, always convolution result before Relu
    :param eps: scalar,
    :param trnFlag: bool, whether training or not
    :param ema_decay: scalar, moving average used of BN's beta and gamma
    :param dtype: tf.dtype, data type
    :return:
        tensor, BN reuslt which has the same shape as _input
    """
    shape = _input.get_shape().as_list()
    with variable_scope.variable_scope(name) as scope:
        beta = variable_scope.get_variable("beta", [shape[-1]], dtype=dtype, initializer=init_ops.constant_initializer(0., dtype=dtype), trainable=True)
        gamma = variable_scope.get_variable("gamma", [shape[-1]], dtype=dtype, initializer=init_ops.random_normal_initializer(1., 0.01, dtype=dtype, seed=20170705), trainable=True)

        if shape.__len__() == 2:  # fc, [batch_size, num_dim]
            batch_mean, batch_var = nn_impl.moments(_input, [0], name="moments")
        elif shape.__len__() == 4:# conv, [batch_size, width, heigth, channel]
            batch_mean, batch_var = nn_impl.moments(_input, [0, 1, 2], name="moments")
        elif shape.__len__() == 5:  # conv, [batch_size, depth, width, heigth, channel]
            batch_mean, batch_var = nn_impl.moments(_input, [0, 1, 2, 3], name="moments")
        else:
            raise 'wrong _input shape, it must have dim of 2 or 4 or 5'

        ema = training.ExponentialMovingAverage(decay=ema_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with ops.control_dependencies([ema_apply_op]):
                return array_ops.identity(batch_mean), array_ops.identity(batch_var)
        mean, var = control_flow_ops.cond(trnFlag, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        bn_out = nn_impl.batch_normalization(_input, mean, var, beta, gamma, eps)
    return bn_out


def _fc(_input, out_dim, name="fc", relu_flag=True, stddev=0.01, dtype=dtypes.float32):
    """
    A wrapped full connection layer used in normal-fc or conv-fc(2D or 3D)
    :param _input: tensor, shape's ndim must be 2(normal) or 4(conv2d) or 5(conv3d)
    :param out_dim: scalar, outout dimension
    :param relu_flag: bool, whether using Relu after fc operation
    :param stddev: scalar, standard deviation used for params' initialization
    :param dtype: tf.dtypes, data type
    :return:
        tensor, shape = [_input.shape[0], out_dim]
    """
    with variable_scope.variable_scope(name) as scope:
        input_shape = _input.get_shape()
        # print 'shape-----------', input_shape
        assert input_shape.ndims == 5 or input_shape.ndims == 4 or input_shape.ndims == 2
        if input_shape.ndims == 2:
            feed_in, dim = (_input, input_shape[-1].value)

        else:
            if input_shape.ndims == 5: # if _input is conv3d result, we need to exchange the axes
                _input = tf.transpose(_input, perm=[0, 1, 4, 2, 3])
            input_shape = _input.get_shape()
            dim = 1
            for dim_id in input_shape[1:].as_list():
                dim *= dim_id
            feed_in = array_ops.reshape(_input, [-1, dim])
        weights = variable_scope.get_variable('weights', shape=[dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype, seed=20170705))
        biases = variable_scope.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0., dtype=dtype))
        act = nn_ops.xw_plus_b(feed_in, weights=weights, biases=biases, name=scope.name)
        if relu_flag:
            return nn_ops.relu(act)
        else:
            return act

def _dropout(_input, keep_prob=0.5, trn_flag=True, name="dropout"):
    """
    A wrapped dropout operation used for CNN
    :param _input: tensor or a list of tensor
    :param keep_prob: scalar, dropout ratio
    :param trn_flag: bool, if it is False, no dropout will be applied
    :return:
        dropout result
    """
    with tf.variable_scope(name) as scope:
        inference = _input
        def apply_dropout():
            if type(inference) in [list, np.array]:
                t_res = []
                for no_care in inference:
                    t_res.append(nn_ops.dropout(no_care, keep_prob=keep_prob))
                return t_res
            else:
                return nn_ops.dropout(inference, keep_prob=keep_prob)
        out = control_flow_ops.cond(trn_flag, apply_dropout, lambda: inference)
    return out



def _mutli_layer_rnn(cell_size, batch_size=None, cell_type='LSTM', num_layers=1, \
                          is_drop_out=True, use_peepholes=True, return_zero_state=False):
    """
    A wrapped rnn structure generator
    **waring:**
        if return_zero_state is True, you must provide 'batch_size' to initialize zeros state
    :param cell_size: scalar, RNN cell size
    :param batch_size: scalar, training batch size
    :param cell_type: string, 'LSTM' or 'GRU'
    :param num_layers: scalar, number of rnn layers used for stacking
    :param is_drop_out: bool, whether to apply dropout operation or not
    :param use_peepholes: bool, whether to use peephole mechanism or not(just for 'LSTM' cell)
    :param return_zero_state: bool, whether to generate the zero state corresponding to rnn structure
    :return:
        a mutli-layer rnn structure
    """
    def _rnn_cell(cell_type, cell_size, is_drop_out=True, _dropout_in=1.0, _dropout_out=0.5, use_peepholes=True):
        cell = cell_type(num_units=cell_size, forget_bias=1.0, use_peepholes=use_peepholes)
        if is_drop_out:
            cell = rnn_cell_impl.DropoutWrapper(cell=cell, input_keep_prob=_dropout_in, output_keep_prob=_dropout_out)
        return cell

    if cell_type == 'LSTM':
        c_type = rnn_cell_impl.LSTMCell
    # elif cell_type == 'GRU':
    #     c_type = rnn_cell_impl.GRUCell
    #     use_peepholes = False
    else:
        raise 'Invalid cell type , try to use "LSTM" or "GRU"'

    cell = [_rnn_cell(c_type, cell_size, is_drop_out=is_drop_out, use_peepholes=use_peepholes) for i in range(num_layers)]
    mutli_layer = rnn_cell_impl.MultiRNNCell(cell)
    if return_zero_state:
        assert batch_size is not None
        return mutli_layer, mutli_layer.zero_state(batch_size)
    return mutli_layer


def _attention_decoder_wrapper(batch_size, num_units, memory, mutli_layer, dtype=dtypes.float32 ,\
                               attention_layer_size=None, cell_input_fn=None, attention_type='B',\
                               probability_fn=None, alignment_history=False, output_attention=True, \
                               initial_cell_state=None, normalization=False, sigmoid_noise=0.,
                               sigmoid_noise_seed=None, score_bias_init=0.):
    """
    A wrapper for rnn-decoder with attention mechanism

    the detail about params explanation can be found at :
        blog.csdn.net/qsczse943062710/article/details/79539005

    :param mutli_layer: a object returned by function _mutli_layer_rnn()

    :param attention_type, string
        'B' is for BahdanauAttention as described in:

          Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
          "Neural Machine Translation by Jointly Learning to Align and Translate."
          ICLR 2015. https://arxiv.org/abs/1409.0473

        'L' is for LuongAttention as described in:

            Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
            "Effective Approaches to Attention-based Neural Machine Translation."
            EMNLP 2015.  https://arxiv.org/abs/1508.04025

        MonotonicAttention is described in :

            Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
            "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
            ICML 2017.  https://arxiv.org/abs/1704.00784

        'BM' :  Monotonic attention mechanism with Bahadanau-style energy function

        'LM' :  Monotonic attention mechanism with Luong-style energy function


        or maybe something user defined in the future
        **warning** :

            if normalization is set True,
            then normalization will be applied to all types of attentions as described in:
                Tim Salimans, Diederik P. Kingma.
                "Weight Normalization: A Simple Reparameterization to Accelerate
                Training of Deep Neural Networks."
                https://arxiv.org/abs/1602.07868

    A example usage:
        att_wrapper, states = _attention_decoder_wrapper(*args)
        while decoding:
            output, states = att_wrapper(input, states)
            ...
            some processing on output
            ...
            input = processed_output
    """

    if attention_type == 'B':
        attention_mechanism = att_w.BahdanauAttention(num_units=num_units, memory=memory,
                                                      probability_fn=probability_fn, normalize=normalization)
    elif attention_type == 'BM':
        attention_mechanism = att_w.BahdanauMonotonicAttention(num_units=num_units, memory=memory,
                                                               normalize=normalization, sigmoid_noise=sigmoid_noise,
                                                               sigmoid_noise_seed=sigmoid_noise_seed, score_bias_init=score_bias_init)
    elif attention_type == 'L':
        attention_mechanism = att_w.LuongAttention(num_units=num_units, memory=memory,
                                                      probability_fn=probability_fn, scale=normalization)
    elif attention_type == 'LM':
        attention_mechanism = att_w.LuongMonotonicAttention(num_units=num_units, memory=memory,
                                                            scale=normalization, sigmoid_noise=sigmoid_noise,
                                                            sigmoid_noise_seed=sigmoid_noise_seed, score_bias_init=score_bias_init)
    else:
        raise 'Invalid attention type'

    att_wrapper = att_w.AttentionWrapper(cell=mutli_layer,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=attention_layer_size,
                                         cell_input_fn=cell_input_fn,
                                         alignment_history=alignment_history,
                                         output_attention=output_attention,
                                         initial_cell_state=initial_cell_state)
    init_states = att_wrapper.zero_state(batch_size=batch_size, dtype=dtype)
    return att_wrapper, init_states


def _eltwise_sum(input_1, input_2, name="eltwise_sum"):
    """
    A element-wise sum operation

    """
    with tf.variable_scope(name) as scope:
        shape1 = input_1.get_shape().as_list()
        shape2 = input_2.get_shape().as_list()
        assert shape1.__len__() == shape2.__len__()
        out = math_ops.add(input_1, input_2)
    return out

# I don't know what this is used for
def _eltwise_sum_conv2d(input_1, input_2, out_channels1=64, out_channels2=256, trnFlag=True, name="eltwise"):
    with tf.variable_scope(name) as scope:
        eltwise = _eltwise_sum(input_1, input_2, name='_eltwise')
        bn1 = _batch_norm(eltwise, trnFlag=trnFlag, name='bn1')
        relu1 = _relu(bn1, name='relu1')
        conv1 = _conv2d(relu1, out_channels1, 1, 1, 1, 1, name='conv1')
        bn2 = _batch_norm(conv1, trnFlag=trnFlag, name='bn2')
        relu2 = _relu(bn2, name='relu2')
        conv2 = _conv2d(relu2, out_channels1, 3, 3, 1, 1, name='conv2')
        bn3 = _batch_norm(conv2, trnFlag=trnFlag, name='bn3')
        relu3 = _relu(bn3, name='relu3')
        conv3 = _conv2d(relu3, out_channels2, 1, 1, 1, 1, name='conv3')
    return relu1, conv3

# maybe something wrong
def _rns(input_1, input_2, out_channels1=96, out_channels2=256, trnFlag=True, name="rns"):
    with tf.variable_scope(name) as scope:
        rns = _eltwise_sum(input_1=input_1, input_2=input_2, name='_rns')
        bn1 = _batch_norm(rns, trnFlag=trnFlag, name='bn1')
        relu1 = _relu(bn1, name='relu1')

        conv2 = _conv2d(relu1, out_channels1, 1, 1, 1, 1, name='conv1')
        bn2 = _batch_norm(conv2, trnFlag=trnFlag, name='bn2')
        relu2 = _relu(bn2, name='relu2')

        conv3 = _conv2d(relu2, out_channels1, 3, 3, 2, 2, name='conv3')
        bn3 = _batch_norm(conv3, trnFlag=trnFlag, name='bn3')
        relu3 = _relu(bn3, name='relu3')

        conv4 = _conv2d(relu3, out_channels2, 1, 1, 1, 1, name='conv4')

        conv5 = _conv2d(relu1, out_channels2, 1, 1, 2, 2, name='conv5')
        bn4 = _batch_norm(conv5, trnFlag=trnFlag, name='bn4')

        return conv4, bn4
