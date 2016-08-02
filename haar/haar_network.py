"""
Author: angles
Date and time: 27/07/16 - 13:26
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from grid_filters import make_grid


def batch_normalization(inputs, decay, epsilon, is_training):
    inputs_shape = inputs.get_shape()
    dtype = inputs.dtype.base_dtype
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    beta = tf.get_variable('beta', shape=params_shape, dtype=dtype, initializer=init_ops.zeros_initializer)
    gamma = tf.get_variable('gamma', shape=params_shape, dtype=dtype, initializer=init_ops.ones_initializer)
    moving_mean = tf.get_variable('moving_mean', shape=params_shape, dtype=dtype,
                                  initializer=init_ops.zeros_initializer, trainable=False)
    moving_variance = tf.get_variable('moving_variance', shape=params_shape, dtype=dtype,
                                      initializer=init_ops.ones_initializer, trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axis, shift=moving_mean)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
        with ops.control_dependencies([update_moving_mean, update_moving_variance]):
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    else:
        outputs = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    return outputs


def convolution(inputs, nb_channels_output, k, s, p):
    dtype = inputs.dtype.base_dtype
    nb_channels_input = utils.last_dimension(inputs.get_shape(), min_rank=4)
    kernel_h, kernel_w = utils.two_element_tuple(k)
    weights_shape = [kernel_h, kernel_w, nb_channels_input, nb_channels_output]
    msr_init_std = np.sqrt(2 / (kernel_h * kernel_w * nb_channels_output))
    weights_init = tf.random_normal(weights_shape, mean=0, stddev=msr_init_std, dtype=dtype)
    weights = tf.get_variable('weights', initializer=weights_init)
    if (nb_channels_input == 1) or (nb_channels_input == 3):
        grid = make_grid(weights)
        tf.image_summary(tf.get_default_graph().unique_name('Filters', mark_as_used=False), grid)
    padded_inputs = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]])
    output = tf.nn.conv2d(padded_inputs, weights, [1, s, s, 1], padding='VALID')
    return output


def linear(name_scope, inputs, nb_output_channels):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_input_channels = utils.last_dimension(inputs.get_shape(), min_rank=2)
        weights_shape = [nb_input_channels, nb_output_channels]
        weights = tf.get_variable('weights', weights_shape, initializer=initializers.xavier_initializer(), dtype=dtype)
        bias = tf.get_variable('bias', [nb_output_channels], initializer=init_ops.zeros_initializer, dtype=dtype)
        output = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
    return output


def layer(name_layer, inputs, nb_channels_output, is_training):
    with tf.variable_scope(name_layer):
        outputs = convolution(inputs, nb_channels_output, 3, 1, 1)
        outputs = batch_normalization(outputs, 0.9, 0.001, is_training)
        outputs = tf.nn.relu(outputs)
    return outputs


def group(name_group, inputs, nb_channels_output, nb_layers, max_pooling, is_training):
    with tf.variable_scope(name_group):
        for idx_layer in range(nb_layers):
            name_layer = 'C_BN_R' + str(idx_layer + 1)
            if idx_layer == 0:
                outputs = layer(name_layer, inputs, nb_channels_output, is_training)
            else:
                outputs = layer(name_layer, outputs, nb_channels_output, is_training)
        if max_pooling:
            outputs = tf.nn.max_pool(outputs, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return outputs


def inference(inputs, is_training):
    outputs = group('group1', inputs, 64, 2, True, is_training)
    outputs = group('group2', outputs, 128, 2, True, is_training)
    outputs = group('group3', outputs, 256, 4, True, is_training)
    outputs = group('group4', outputs, 512, 4, True, is_training)
    outputs = group('group5', outputs, 512, 4, False, is_training)
    outputs = tf.nn.avg_pool(outputs, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')
    outputs = tf.squeeze(outputs, squeeze_dims=[1, 2])
    outputs = linear('linear', outputs, 10)
    return outputs


"""
inputs = tf.placeholder(tf.float32, [128, 32, 32, 3])
outputs = inference(inputs, is_training=True)
"""
