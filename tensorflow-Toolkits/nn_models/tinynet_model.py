#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
from tensorflow.python.ops import variable_scope

__all__ = ['tinynet']

def tinynet(frame_imgs,  trn_Flag, keep_prob=0.5, out_num=10, return_fea_map=False, name='tinynet'):
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_height, target_width, img_channel])
    with variable_scope.variable_scope(name):
        conv_pre = _conv2d(frame_imgs, 8, 5, 5, 2, 2, stddev=0.07142857142857142, name='tiny_conv_pre')
        bn_pre = _batch_norm(conv_pre, trnFlag=trn_Flag, name='tiny_bn_pre')
        relu_pre = _relu(bn_pre, name='tiny_relu_pre')
        max_pre = _max_pool_2d(relu_pre, 2, 2, sh=2, sw=2, name='tiny_pool_pre')

        conv1_1 = _conv2d(max_pre, 16, 3, 3, 1, 1, stddev=0.11785113019775792, name='tiny_conv1_1')
        bn_conv1_1 = _batch_norm(conv1_1, trnFlag=trn_Flag, name='tiny_bn_conv1_1')
        relu1_1 = _relu(bn_conv1_1, name='tiny_relu1_1')

        conv1_2 = _conv2d(relu1_1, 16, 3, 3, 1, 1, stddev=0.11785113019775792, name='tiny_conv1_2')
        bn_conv1_2 = _batch_norm(conv1_2, trnFlag=trn_Flag, name='tiny_bn_conv1_2')
        relu1_2 = _relu(bn_conv1_2, name='tiny_relu1_2')
        max1_2 = _max_pool_2d(relu1_2, 2, 2, sh=2, sw=2, name='tiny_pool1_2')

        conv2_1 = _conv2d(max1_2, 24, 3, 3, 1, 1, stddev=0.09622504486493763, name='tiny_conv2_1')
        bn_conv2_1 = _batch_norm(conv2_1, trnFlag=trn_Flag, name='tiny_bn_conv2_1')
        relu2_1 = _relu(bn_conv2_1, name='tiny_relu2_1')

        conv2_2 = _conv2d(relu2_1, 24, 3, 3, 1, 1, stddev=0.09622504486493763, name='tiny_conv2_2')
        bn_conv2_2 = _batch_norm(conv2_2, trnFlag=trn_Flag, name='tiny_bn_conv2_2')
        relu2_2 = _relu(bn_conv2_2, name='tiny_relu2_2')
        max2_2 = _max_pool_2d(relu2_2, 2, 2, sh=2, sw=2, name='tiny_pool2_2')

        conv3_1 = _conv2d(max2_2, 80, 3, 3, 1, 1,  stddev=0.08333333333333333, name='tiny_conv3_1')
        bn_conv3_1 = _batch_norm(conv3_1, trnFlag=trn_Flag, name='tiny_bn_conv3_1')
        relu3_1 = _relu(bn_conv3_1, name='tiny_relu3_1')

        conv3_2 = _conv2d(relu3_1, 160, 3, 3, 1, 1, stddev=0.08333333333333333, name='tiny_conv3_2')
        bn_conv3_2 = _batch_norm(conv3_2, trnFlag=trn_Flag, name='tiny_bn_conv3_2')
        relu3_2 = _relu(bn_conv3_2, name='tiny_relu3_2')
        if return_fea_map:
            relu3_2
        fc1 = _fc(relu3_2, 4096, name='tiny_fc1', relu_flag=False, stddev=0.001)
        bn_fc1 = _batch_norm(fc1, trnFlag=trn_Flag, name='tiny_bn_fc1')
        relu_fc1 = _relu(bn_fc1, name='tiny_relu_fc1')

        fc2 = _fc(relu_fc1, out_num, name='tiny_fc2', relu_flag=False, stddev=0.001)
        bn_fc2 = _batch_norm(fc2, trnFlag=trn_Flag, name='tiny_bn_fc2')
        relu_fc2 = _relu(bn_fc2, name='tiny_relu_fc2')

        # fc3 = fc(relu_fc2, out_num, relu_flag=False)

        return relu_fc2
