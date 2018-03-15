#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
__all__ = ['sg', 'sg_full']
def sg_full(frame_imgs, trn_Flag, keep_prob=0.5, out_channels=10, return_fea_map=True):
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_height, target_width,  img_channel])

    conv1 = _conv2d(frame_imgs, 48, 5, 5, 2, 2, name='sg_conv1')
    bn1 = _batch_norm(conv1, trnFlag=trn_Flag, name='sg_conv1_bn')
    relu1 = _relu(bn1, name='sg_conv1_relu')
    max1 = _max_pool_2d(relu1, 3, 3, 2, 2, name='sg_conv1_pool')

    conv2_a = _conv2d(max1, 96, 1, 1, 1, 1, name='sg_conv2_a')
    bn2_a = _batch_norm(conv2_a, trnFlag=trn_Flag, name='sg_conv2_a_bn')
    relu2_a = _relu(bn2_a, name='sg_conv2_a_relu')

    conv2 = _conv2d(relu2_a, 96, 3, 3, 1, 1, name='sg_conv2')
    bn2 = _batch_norm(conv2, trnFlag=trn_Flag, name='sg_conv2_bn')
    relu2 = _relu(bn2, name='sg_conv2_relu')

    conv3_a = _conv2d(relu2, 96, 1, 1, 1, 1, name='sg_conv3_a')
    bn3_a = _batch_norm(conv3_a, trnFlag=trn_Flag, name='sg_conv3_a_bn')
    relu3_a = _relu(bn3_a, name='sg_conv3_a_relu')

    conv3 = _conv2d(relu3_a, 96, 3, 3, 1, 1, name='sg_conv3')
    bn3 = _batch_norm(conv3, trnFlag=trn_Flag, name='sg_conv3_bn')
    relu3 = _relu(bn3, name='sg_conv3_relu')
    max3 = _max_pool_2d(relu3, 3, 3, 2, 2, name='sg_conv3_pool')

    conv4_a = _conv2d(max3, 192, 1, 1, 1, 1, name='sg_conv4_a')
    bn4_a = _batch_norm(conv4_a, trnFlag=trn_Flag, name='sg_conv4_a_bn')
    relu4_a = _relu(bn4_a, name='sg_conv4_a_relu')

    conv4 = _conv2d(relu4_a, 192, 3, 3, 1, 1, name='sg_conv4')
    bn4 = _batch_norm(conv4, trnFlag=trn_Flag, name='sg_conv4_bn')
    relu4 = _relu(bn4, name='sg_conv4_relu')

    conv5_a = _conv2d(relu4, 192, 1, 1, 1, 1, name='sg_conv5_a')
    bn5_a = _batch_norm(conv5_a, trnFlag=trn_Flag, name='sg_conv5_a_bn')
    relu5_a = _relu(bn5_a, name='sg_conv5_a_relu')

    conv5 = _conv2d(relu5_a, 192, 3, 3, 1, 1, name='sg_conv5')
    bn5 = _batch_norm(conv5, trnFlag=trn_Flag, name='sg_conv5_bn')
    relu5 = _relu(bn5, name='sg_conv5_relu')

    conv6_a = _conv2d(relu5, 192, 1, 1, 1, 1, name='sg_conv6_a')
    bn6_a = _batch_norm(conv6_a, trnFlag=trn_Flag, name='sg_conv6_a_bn')
    relu6_a = _relu(bn6_a, name='sg_conv6_a_relu')

    conv6 = _conv2d(relu6_a, 192, 3, 3, 1, 1, name='sg_conv6')
    bn6 = _batch_norm(conv6, trnFlag=trn_Flag, name='sg_conv6_bn')
    relu6 = _relu(bn6, name='sg_conv6_relu')
    max6 = _max_pool_2d(relu6, 3, 3, 2, 2, name='sg_conv6_pool')

    fc1 = _fc(max6, 4096, relu_flag=False, name='sg_fc1')
    bn_fc1 = _batch_norm(fc1, trnFlag=trn_Flag, name='sg_fc1_bn')
    relu_fc1 = _relu(bn_fc1, name='sg_fc1_relu')
    drop_fc1 = _dropout(relu_fc1, keep_prob=keep_prob, trn_flag=trn_Flag, name='sg_fc1_drop')

    fc2 = _fc(drop_fc1, out_channels, relu_flag=False, name='sg_fc2')
    bn_fc2 = _batch_norm(fc2, trnFlag=trn_Flag, name='sg_fc2_bn')
    relu_fc2 = _relu(bn_fc2, name='sg_fc2_relu')
    drop_fc2 = _dropout(relu_fc2, keep_prob=keep_prob, trn_flag=trn_Flag, name='sg_fc2_drop')

    fc3 = _fc(drop_fc2, out_channels, relu_flag=False, name='sg_fc3')

    return relu_fc2


def sg(frame_imgs, trn_Flag, keep_prob=0.5, out_channels=10, return_fea_map=True):
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_height, target_width,  img_channel])

    conv1 = _conv2d(frame_imgs, 48, 5, 5, 2, 2, name='sg_conv1')
    bn1 = _batch_norm(conv1, trnFlag=trn_Flag, name='sg_conv1_bn')
    relu1 = _relu(bn1, name='sg_conv1_relu')
    max1 = _max_pool_2d(relu1, 3, 3, 2, 2, name='sg_conv1_pool')

    conv2_a = _conv2d(max1, 96, 1, 1, 1, 1, name='sg_conv2_a')
    bn2_a = _batch_norm(conv2_a, trnFlag=trn_Flag, name='sg_conv2_a_bn')
    relu2_a = _relu(bn2_a, name='sg_conv2_a_relu')

    conv2 = _conv2d(relu2_a, 96, 3, 3, 1, 1, name='sg_conv2')
    bn2 = _batch_norm(conv2, trnFlag=trn_Flag, name='sg_conv2_bn')
    relu2 = _relu(bn2, name='sg_conv2_relu')

    conv3_a = _conv2d(relu2, 96, 1, 1, 1, 1, name='sg_conv3_a')
    bn3_a = _batch_norm(conv3_a, trnFlag=trn_Flag, name='sg_conv3_a_bn')
    relu3_a = _relu(bn3_a, name='sg_conv3_a_relu')

    conv3 = _conv2d(relu3_a, 96, 3, 3, 1, 1, name='sg_conv3')
    bn3 = _batch_norm(conv3, trnFlag=trn_Flag, name='sg_conv3_bn')
    relu3 = _relu(bn3, name='sg_conv3_relu')
    max3 = _max_pool_2d(relu3, 3, 3, 2, 2, name='sg_conv3_pool')

    conv4_a = _conv2d(max3, 192, 1, 1, 1, 1, name='sg_conv4_a')
    bn4_a = _batch_norm(conv4_a, trnFlag=trn_Flag, name='sg_conv4_a_bn')
    relu4_a = _relu(bn4_a, name='sg_conv4_a_relu')

    conv4 = _conv2d(relu4_a, 192, 3, 3, 1, 1, name='sg_conv4')
    bn4 = _batch_norm(conv4, trnFlag=trn_Flag, name='sg_conv4_bn')
    relu4 = _relu(bn4, name='sg_conv4_relu')

    conv5_a = _conv2d(relu4, 192, 1, 1, 1, 1, name='sg_conv5_a')
    bn5_a = _batch_norm(conv5_a, trnFlag=trn_Flag, name='sg_conv5_a_bn')
    relu5_a = _relu(bn5_a, name='sg_conv5_a_relu')

    conv5 = _conv2d(relu5_a, 192, 3, 3, 1, 1, name='sg_conv5')
    bn5 = _batch_norm(conv5, trnFlag=trn_Flag, name='sg_conv5_bn')
    relu5 = _relu(bn5, name='sg_conv5_relu')

    conv6_a = _conv2d(relu5, 192, 1, 1, 1, 1, name='sg_conv6_a')
    bn6_a = _batch_norm(conv6_a, trnFlag=trn_Flag, name='sg_conv6_a_bn')
    relu6_a = _relu(bn6_a, name='sg_conv6_a_relu')

    conv6 = _conv2d(relu6_a, 192, 3, 3, 1, 1, name='sg_conv6')
    bn6 = _batch_norm(conv6, trnFlag=trn_Flag, name='sg_conv6_bn')
    relu6 = _relu(bn6, name='sg_conv6_relu')
    max6 = _max_pool_2d(relu6, 3, 3, 2, 2, name='sg_conv6_pool')

    if return_fea_map:
        return max6

    fc1 = _fc(max6, 4096, relu_flag=False, name='sg_fc1')
    bn_fc1 = _batch_norm(fc1, trnFlag=trn_Flag, name='sg_fc1_bn')
    relu_fc1 = _relu(bn_fc1, name='sg_fc1_relu')
    drop_fc1 = _dropout(relu_fc1, keep_prob=keep_prob, trn_flag=trn_Flag, name='sg_fc1_drop')

    fc2 = _fc(drop_fc1, out_channels, relu_flag=False, name='sg_fc2')
    bn_fc2 = _batch_norm(fc2, trnFlag=trn_Flag, name='sg_fc2_bn')
    relu_fc2 = _relu(bn_fc2, name='sg_fc2_relu')
    # drop_fc2 = dropout(relu_fc2, keep_prob=keep_prob, trn_flag=trn_Flag, name='sg_fc2_drop')
    #
    # fc3 = fc(drop_fc2, out_num, relu_flag=False, name='sg_fc3')

    return relu_fc2