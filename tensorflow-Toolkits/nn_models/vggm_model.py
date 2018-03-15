#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
from utils.array_ops import *
__all__ = ['vggm']
def vggm(frame_imgs, trn_Flag, keep_prob=0.5, out_channels=10, return_fea_map=True):
    conv2 = _conv2d(frame_imgs, 96, 1, 1, 1, 1, name='vggm_conv2')
    bn2 = _batch_norm(conv2, trnFlag=trn_Flag, name='vggm_bn2')
    relu2 = _relu(bn2, name='vggm_relu2')
    pool2 = _max_pool_2d(relu2, 3, 3, 2, 2, name='vggm_pool2')

    conv3 = _conv2d(pool2, 256, 3, 3, 1, 1, name='vggm_conv3')
    bn3 = _batch_norm(conv3, trnFlag=trn_Flag, name='vggm_bn3')
    relu3 = _relu(bn3, name='vggm_relu3')
    pool3 = _max_pool_2d(relu3, 3, 3, 2, 2, name='vggm_pool3')

    conv4 = _conv2d(pool3, 512, 3, 3, 1, 1, name='vggm_conv4')
    bn4 = _batch_norm(conv4, trnFlag=trn_Flag, name='vggm_bn4')
    relu4 = _relu(bn4, name='vggm_relu4')

    conv5 = _conv2d(relu4, 512, 3, 3, 1, 1, name='vggm_conv5')
    bn5 = _batch_norm(conv5, trnFlag=trn_Flag, name='vggm_bn5')
    relu5 = _relu(bn5, name='vggm_relu5')

    conv6 = _conv2d(relu5, 512, 3, 3, 1, 1, name='vggm_conv6')
    bn6 = _batch_norm(conv6, trnFlag=trn_Flag, name='vggm_bn6')
    relu6 = _relu(bn6, name='vggm_relu6')
    pool6 = _max_pool_2d(relu6, 3, 3, 2, 2, name='vggm_pool6')
    if return_fea_map:
        return pool6
    fc1 = _fc(pool6, 4096, name="vggm_fc1")
    bn_fc1 = _batch_norm(fc1, trnFlag=trn_Flag, name='vggm_bn_fc1')
    relu_fc1 = _relu(bn_fc1, name='vggm_relu_fc1')

    fc2 = _fc(relu_fc1, out_channels, name="vggm_fc2", relu_flag=False)
    bn_fc2 = _batch_norm(fc2, trnFlag=trn_Flag, name='vggm_bn_fc2')
    relu_fc2 = _relu(bn_fc2, name='vggm_relu_fc2')

    # fc3 = fc(relu_fc2, out_num, name='vggm_fc3', relu_flag=False)

    return relu_fc2