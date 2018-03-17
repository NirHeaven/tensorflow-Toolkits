#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
from tensorflow.python.ops import variable_scope

__all__ = ['lenet']

def lenet(_input, trn_Flag, keep_prob=0.5, out_channels=10, return_fea_map=False, name='lanet'):
    # target_heigth =28
    # target_width = 28
    # img_channel = 1
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_width, target_height, img_channel])
    with variable_scope.variable_scope(name):
        conv1 = _conv2d(_input, 20, 5, 5, 1, 1, name='lenet_conv1')
        bn1 = _batch_norm(conv1, trnFlag=trn_Flag, name='lenet_bn1')
        # bn1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=trn_Flag, scope='lenet_bn1')
        max1 = _max_pool_2d(bn1, 2, 2, 2, 2, name='lenet_conv1_pool')

        conv2 = _conv2d(max1, 50, 5, 5, 1, 1, name='lenet_conv2')
        bn2 = _batch_norm(conv2, trnFlag=trn_Flag, name='lenet_bn2')
        # bn2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=trn_Flag, scope='lenet_bn2')
        max1 = _max_pool_2d(bn2, 2, 2, 2, 2, name='lenet_conv2_pool')
        if return_fea_map:
            return max1
        fc1 = _fc(max1, out_channels, name='lenet_fc1', relu_flag=True)
        bn3 = _batch_norm(fc1, trnFlag=trn_Flag, name='lenet_bn3')
        # bn3 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=trn_Flag, scope='lenet_bn3')
        # fc2 = fc(bn3, out_num, name='lenet_fc2', relu_flag=False)
        return bn3