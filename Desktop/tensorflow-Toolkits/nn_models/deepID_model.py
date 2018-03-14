#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
from utils.array_ops import *

def deepID(frame_imgs, trn_Flag=True, keep_prob=0.5, out_channels=10, target_width=64, target_height=64, img_channel=3):
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_height, target_width, img_channel])

    conv1 = _conv2d(frame_imgs, 20, 5, 5, 1, 1, name='deepID_conv1')
    bn1 = _batch_norm(conv1, trnFlag=trn_Flag, name='deepID_bn1')
    relu1 = _relu(bn1, name='deepID_relu1')
    max1 = _max_pool_2d(relu1, 2, 2, 2, 2, name='deepID_pool1')

    conv2 = _conv2d(max1, 40, 3, 3, 1, 1, name='deepID_conv2')
    bn2 = _batch_norm(conv2, trnFlag=trn_Flag, name='deepID_bn2')
    relu2 = _relu(bn2, name='deepID_relu2')
    max2 = _max_pool_2d(relu2, 2, 2, 2, 2, name='deepID_pool2')

    conv3 = _conv2d(max2, 60, 3, 3, 1, 1, name='deepID_conv3')
    bn3 = _batch_norm(conv3, trnFlag=trn_Flag, name='deepID_bn3')
    relu3 = _relu(bn3, name='deepID_relu3')
    max3 = _max_pool_2d(relu3, 2, 2, 2, 2, name='deepID_pool3')
    flat3 = _flattern(max3, axis=0, name='deepID_flat3')

    conv4 = _conv2d(max3, 80, 3, 3, 2, 2, name='deepID_conv4')
    bn4 = _batch_norm(conv4, trnFlag=trn_Flag, name='deepID_bn4')
    relu4 = _relu(bn4, name='deepID_relu4')
    flat4 = _flattern(relu4, axis=0, name='deepID_flat4')

    concat_34 = _concat(flat3, flat4, axis=1, name='deepID_concat34')

    fc1 = _fc(concat_34, 1024, name='deepID_fc1', relu_flag=False)
    bn_fc1 = _batch_norm(fc1, trnFlag=trn_Flag,  name='deepID_bn_fc1')
    relu_fc1 = _relu(bn_fc1, name='deepID_relu_fc1')
    drop_fc1 = _dropout(relu_fc1, keep_prob=keep_prob, trn_flag=trn_Flag, name='deepID_drop_fc1')

    fc2 = _fc(drop_fc1, out_channels, name='deepID_fc2', relu_flag=False)
    bn_fc2 = _batch_norm(fc2, trnFlag=trn_Flag,  name='deepID_bn_fc2')
    relu_fc2 = _relu(bn_fc2, name='deepID_relu_fc2')
    drop_fc2 = _dropout(relu_fc2, keep_prob=keep_prob, trn_flag=trn_Flag, name='deepID_drop_fc2')

    # fc3 = fc(drop_fc2, num_out=out_num, relu_flag=False, name='deepID_fc3')

    return drop_fc2