#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from nn_models import *
from utils import *
import tensorflow as tf


class model_assemble(object):

    def __init__(self, layers_list, params, logit_shape=None):
        """

        :param logit_shape:
            a list tells final output's shape, allow one -1 in the shape list
        :param layers_list:
            a list that contains every layer operation to assemble the nn
            Example:
                [resnet80, _conv2d, _max_pool2d, _relu, ＿fc]
        :param params:
            1.a dictionary that contains correspond params to the layers in 'layers_list',
            2.whoes keys are 'component_1',...,'component_n', n is the length of 'layers_list'
            3.every value in the dictionary is also a dictionary

            **warning**:
                except for the first component,there is no need to provide '_input' parameter,
                the output of the previous component will be feed to the next component automatically.
                but if you need a reshape operation before feeding the output to next component,
                you should provide a "shape": [...] key-value in the dic,
                if there is no params, then you just need to provide a empty dictionary

            Example:
                layers_list = [resnet80, _conv2d, _conv2d, relu]
                params = {
                    'component_1':{
                        '_input': tf.placeholder(...),
                        'trn_flag': ...,
                    },
                    'component_2':{
                        'out_channels': ...,
                        'kd': ...,
                    },
                    'component_3':{
                        'shape': ...,
                        'out_channels': ...,
                        'kd': ...,
                    },
                     'component_4':{

                    }

                }
        """

        assert len(layers_list) == len(params)

        self._layers_list = layers_list
        self._params = params
        self._logit_shape = logit_shape



    def inference(self):
        f_output = None
        for c, component in enumerate(self._layers_list):
            if c == 0:
                f_output = component(**self._params['component_' + str(c + 1)])

            else:
                if 'shape' in self._params['component_' + str(c + 1)]:
                    f_output = tf.reshape(f_output, self._params['component_' + str(c + 1)]['shape'])
                    self._params['component_' + str(c + 1)].pop('shape')
                f_output = component(f_output, **self._params['component_' + str(c + 1)])
        if self._logit_shape:
            self._logits = tf.reshape(f_output, self._logit_shape)
        else:
            self._logits = f_output

# test
if __name__ == '__main__':
    l = [layer_ops._conv2d, deepID_model.deepID, layer_ops._relu]
    params = {
        'component_1': {
            '_input': tf.placeholder(tf.float32, (20, 80, 80, 3)),
            'out_channels': 512
        },
        'component_2': {
            'shape': [40, 40, 20, 512],
            'trn_Flag': tf.placeholder(tf.bool)
        },
        'component_3': {

        }
    }

    m = model_assemble(l, params, [2, -1, 5])
    m.inference()
