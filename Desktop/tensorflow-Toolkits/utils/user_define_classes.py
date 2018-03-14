#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2018.3.14;
# Copyright (C) 2017 Shuang Yang, Mingmin Yang /@

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism


class UserDefinedAttention(_BaseAttentionMechanism):
    """
    A user defined attention mechaniam to achieve different energy-score function
    all the params are the same as BahdanauAttention or ...
    except the following three params:
    :param  use_query_layer, bool,
        whether to apply a linear transform(s.t. matmul operation) to query before calculate score
    :param  use_memory_layer, bool,
        whether to apply a linear transform(s.t. matmul operation) to memory before calculate score
    :param score_function, callable,
        how to calculate energy score.
        **warning**:
            it must accept three params : query, keys, scale
            and return a tensor with shape [memory.shape[0], memory.shape[1]], called alignment
        if use_query_layer is False, then query is always rnn's hidden state, else query = w1 * query
        if use_memory_layer is False, then keys are just the memory, else keys = w2 * memory

    """
    def __init__(self,
                num_units,
                memory,
                score_function,
                use_query_layer=True,
                use_memory_layer=True,
                memory_sequence_length=None,
                normalize=False,
                probability_fn=None,
                score_mask_value=float("-inf"),
                name="UserDefinedAttention"):

        assert score_function is not None
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(UserDefinedAttention, self).__init__(
            query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False) if use_query_layer else None,
            memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_function = score_function
        self._use_memory_layer = use_memory_layer

    def __call__(self, query, previous_alignments):

        with variable_scope.variable_scope(None, "userdefined_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            if self._use_memory_layer:
                score = self._score_function(processed_query, self._keys, self._normalize)
            else:
                keys = self._values
                score = self._score_function(processed_query, keys, self._normalize)
            alignments = self._probability_fn(score, previous_alignments)
        return alignments