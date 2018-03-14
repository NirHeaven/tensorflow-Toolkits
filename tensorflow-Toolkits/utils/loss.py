#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes


def _center_loss(logit, labels, alpha, lam, num_classes, dtype=dtypes.float32):
    """
    coumpute the center loss and update the centers,
    followed by 'A Discriminative Feature Learning Approach for Deep Face Recognition',ECCV 2016

    :param logit: output of NN full connection layer, [batch_size, feature_dimension] tensor
    :param labels: true label of every sample, [batch_size] tensor without ont-hot
    :param alpha: learning rate about speed of updating, 0-1 float
    :param lam: center loss weight compared to softmax loss and others
    :param num_classes: classes numbers,int
    :return:
        loss: the computed center loss
        centers: tensor of all centers,[num_classes, feature_dimension]
        centers_update_op: should be running while training the model to update centers
    """

    # get feature dimension
    fea_dimension = array_ops.shape(logit)[1]

    # initialize centers
    centers = variable_scope.get_variable('centers', [num_classes, fea_dimension], dtype=dtype,
                              initializer=init_ops.constant_initializer(0), trainable=False)

    labels = array_ops.reshape(labels, [-1])

    # get centers about current batch
    centers_batch = array_ops.gather(centers, labels)

    # compote l2 loss
    loss = nn_ops.l2_loss(logit - centers_batch) * lam

    # compute the difference between each sample and their corresponding center
    diff = centers_batch - logit

    # compute delta of corresponding center
    unique_label, unique_idx, unique_count = array_ops.unique_with_counts(labels)
    appear_times = array_ops.gather(unique_count, unique_idx)
    appear_times = array_ops.reshape(appear_times, [-1, 1])
    delta_centers = diff / math_ops.cast(1 + appear_times, tf.float32)
    delta_centers = delta_centers * alpha

    # update centers
    center_update_op = state_ops.scatter_sub(centers, labels, delta_centers)

    return loss, centers, center_update_op