#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from utils.layer_ops import *
from tensorflow.python.ops import variable_scope

__all__ = ['resnet80']

def resnet80(frame_imgs, trn_Flag, keep_prob=0.5, out_channels=10, return_fea_map=False, name='resnet80'):
    # input_layer = tf.reshape(frame_imgs, shape=[-1, target_height, target_width, img_channel])
    with variable_scope.variable_scope(name):
        bn1 = _batch_norm(frame_imgs, trnFlag=trn_Flag, name='resnet80_bn1')

        conv_pre = _conv2d(bn1, 32, 7, 7, 2, 2, name='resnet80_conv')
        max_pre = _max_pool_2d(conv_pre, 3, 3, sh=2, sw=2, name='resnet80_pool_pre')
        bn2 = _batch_norm(max_pre, trnFlag=trn_Flag, name='resnet80_bn2')
        relu1 = _relu(bn2, name='resnet80_relu1')
        conv1 = _conv2d(relu1, 16, 1, 1, 1, 1, name='resnet80_conv1')
        bn3 = _batch_norm(conv1, trnFlag=trn_Flag, name='resnet80_bn3')
        relu2 = _relu(bn3, name='resnet80_relu2')
        conv2 = _conv2d(relu2, 16, 3, 3, 1, 1, name='resnet80_conv2')
        bn4 = _batch_norm(conv2, trnFlag=trn_Flag, name='resnet80_bn4')
        relu3 = _relu(bn4, name='resnet80_relu3')
        conv3 = _conv2d(relu3, 64, 1, 1, 1, 1, name='resnet80_conv3')
        conv4 = _conv2d(relu1, 64, 1, 1, 1, 1, name='resnet80_conv4')
        bn5 = _batch_norm(conv4, trnFlag=trn_Flag, name='resnet80_bn5')

        relu4, conv7 = _eltwise_sum_conv2d(input_1=conv3, input_2=bn5, trnFlag=trn_Flag, out_channels1=16, out_channels2=64, name='resnet80_eltwise1')

        relu7, conv10 = _eltwise_sum_conv2d(input_1=relu4, input_2=conv7, trnFlag=trn_Flag, out_channels1=16, out_channels2=64, name='resnet80_eltwise2')

        conv13, bn15 = _rns(input_1=relu7, input_2=conv10, out_channels1=32, out_channels2=128, trnFlag=trn_Flag, name='resnet80_rnsA')

        relu13, conv17 = _eltwise_sum_conv2d(input_1=conv13, input_2=bn15, out_channels1=32, out_channels2=128, trnFlag=trn_Flag, name='resnet80_eltwise3')

        relu16, conv20 = _eltwise_sum_conv2d(input_1=relu13, input_2=conv17, out_channels1=32, out_channels2=128, trnFlag=trn_Flag, name='resnet80_eltwise4')

        relu19, conv23 = _eltwise_sum_conv2d(input_1=relu16, input_2=conv20, out_channels1=32, out_channels2=128, trnFlag=trn_Flag, name='resnet80_eltwise5')

        conv26, bn28 = _rns(input_1=relu19, input_2=conv23, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_rnsB')

        relu25, conv30 = _eltwise_sum_conv2d(input_1=conv26, input_2=bn28, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise6')

        relu28, conv33 = _eltwise_sum_conv2d(input_1=relu25, input_2=conv30, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise7')

        relu31, conv36 = _eltwise_sum_conv2d(input_1=relu28, input_2=conv33, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise8')

        relu34, conv39 = _eltwise_sum_conv2d(input_1=relu31, input_2=conv36, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise9')

        relu37, conv42 = _eltwise_sum_conv2d(input_1=relu34, input_2=conv39, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise10')

        relu40, conv45 = _eltwise_sum_conv2d(input_1=relu37, input_2=conv42, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise11')

        relu43, conv48 = _eltwise_sum_conv2d(input_1=relu40, input_2=conv45, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise12')

        relu46, conv51 = _eltwise_sum_conv2d(input_1=relu43, input_2=conv48, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise13')

        relu49, conv54 = _eltwise_sum_conv2d(input_1=relu46, input_2=conv51, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise14')

        relu52, conv57 = _eltwise_sum_conv2d(input_1=relu49, input_2=conv54, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise15')

        relu55, conv60 = _eltwise_sum_conv2d(input_1=relu52, input_2=conv57, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise16')

        relu58, conv63 = _eltwise_sum_conv2d(input_1=relu55, input_2=conv60, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise17')

        relu61, conv66 = _eltwise_sum_conv2d(input_1=relu58, input_2=conv63, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise18')

        relu64, conv69 = _eltwise_sum_conv2d(input_1=relu61, input_2=conv66, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise19')

        relu67, conv72 = _eltwise_sum_conv2d(input_1=relu64, input_2=conv69, out_channels1=64, out_channels2=256, trnFlag=trn_Flag, name='resnet80_eltwise20')

        conv75, bn77 = _rns(input_1=relu67, input_2=conv72, out_channels1=96, out_channels2=384, trnFlag=trn_Flag, name='resnet80_rnsC')

        relu73, conv79 = _eltwise_sum_conv2d(input_1=conv75, input_2=bn77, out_channels1=96, out_channels2=384, trnFlag=trn_Flag, name='resnet80_eltwise21')

        relu76, conv82 = _eltwise_sum_conv2d(input_1=relu73, input_2=conv79, out_channels1=96, out_channels2=384, trnFlag=trn_Flag, name='resnet80_eltwise22')

        d_rns = _eltwise_sum(relu76, conv82, name='resnet80_rnsD')

        d_pool = _max_pool_2d(d_rns, 3, 3, sh=2, sw=2, name='resnet80_d_pool')
        bn84 = _batch_norm(d_pool, trnFlag=trn_Flag, name='resnet80_bn84')
        relu79 = _relu(bn84, name='resnet80_relu79')
        if return_fea_map:
            return relu79
        fc1 = _fc(relu79, out_channels, relu_flag=False, name='resnet80_fc1')
        bn85 = _batch_norm(fc1, trnFlag=trn_Flag, name='resnet80_bn85')
        relu80 = _relu(bn85, name='resnet80_relu80')

        # fc2 = fc(relu80, num_out=out_num, relu_flag=False, name='resnet80_fc2')

        return relu80