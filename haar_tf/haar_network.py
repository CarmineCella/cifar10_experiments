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
from haar import haar_and_1x1_relu, marginal_2d_conv

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

def linear(name_scope, inputs, nb_output_channels):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_input_channels = utils.last_dimension(inputs.get_shape(), min_rank=2)
        weights_shape = [nb_input_channels, nb_output_channels]
        weights = tf.get_variable('weights', weights_shape, initializer=initializers.xavier_initializer(), dtype=dtype)
        bias = tf.get_variable('bias', [nb_output_channels], initializer=init_ops.zeros_initializer, dtype=dtype)
        output = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
    return output


def conv_bn_relu(scope_name, inp, n_output_channels, is_training, kernel_size=5,
                 wd=1e-4, strides=(1,1),
                 batch_norm=True):
  with tf.variable_scope(scope_name) as scope:
    n_input_channels = inp.get_shape()[3]
    kernel = tf.get_variable(
        'weights', shape=(kernel_size, kernel_size,
                          n_input_channels, n_output_channels),
        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd)
        tf.add_to_collection('losses', weight_decay)

    conv = tf.nn.conv2d(inp, kernel, (1,) + strides + (1,),  padding='SAME')
    biases = tf.get_variable('biases', (n_output_channels,),
                              initializer=tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    if batch_norm:
        bn = tf.contrib.layers.batch_norm(bias, is_training=is_training)
        conv1 = tf.nn.relu(bn)
    else:
        conv1 = tf.nn.relu(bias)
    return conv1

def marginal_bn_relu(scope_name, inp, n_output_channels, is_training, kernel_size=3,
                 wd=1e-4,
                 batch_norm=True):
  with tf.variable_scope(scope_name) as scope:
    kernel = tf.get_variable(
        'weights', shape=(kernel_size, kernel_size,
                          1, n_output_channels),
        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd)
        tf.add_to_collection('losses', weight_decay)

    conv = marginal_2d_conv(inp, kernel)
    n_input_channels = inp.get_shape()[3].value
    biases = tf.get_variable('biases', (n_output_channels * n_input_channels,),
                              initializer=tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    if batch_norm:
        bn = tf.contrib.layers.batch_norm(bias, is_training=is_training)
        conv1 = tf.nn.relu(bn)
    else:
        conv1 = tf.nn.relu(bias)
    return conv1

def inference(inputs, is_training, batch_norm=False):
    # 128 x 32 x 32 x 3

    conv1 = conv_bn_relu('conv1', inputs, 64, is_training=is_training, batch_norm=batch_norm,
                         kernel_size=5, wd=1e-4)

    # 128 x 32 x 32 x 64
    
    l1 = haar_and_1x1_relu(conv1, 64, scope_name='haar1', concat_axis=3,
                          is_training=is_training, batch_norm=batch_norm,
                          input_shape=(128, 32, 32, 64))
    
    # 128 x 16 x 16 x 64  channels at the end
    
    l2 = haar_and_1x1_relu(l1, 64, scope_name='haar2', concat_axis=3,
                          is_training=is_training, batch_norm=batch_norm,
                          input_shape=(128, 16, 16, 64))
    
    # 128 x 8 x 8 x 64

    #l2_pooled = tf.reduce_mean(l2, 
    l3 = haar_and_1x1_relu(l2, 64, scope_name='haar3', concat_axis=3,
                           is_training=is_training, batch_norm=batch_norm,
                           input_shape=(128, 8, 8, 64))
    # 128 x 4 x 4 x 64
    
    flattened = tf.reshape(l3, (128, 1024))
    lin1 = linear('lin1', flattened, 512)
    lin2 = linear('lin2', tf.nn.relu(lin1), 10)
    return lin2


def inference_perceptron(inputs, is_training, batch_norm=False):
    # 128 x 32 x 32 x 3

    inputs_bw = tf.reduce_mean(inputs, reduction_indices=3)

    dropout_keep_prob = .4 if is_training else 1.
    
    flattened = tf.reshape(inputs_bw, (128, 1024))
    lin1 = linear('lin1', flattened, 1024)
    lin2 = linear('lin2', tf.nn.relu(lin1), 1024)
    lin3 = linear('lin3', tf.nn.relu(lin2), 1024)
    lin4 = linear('lin4', tf.nn.dropout(tf.nn.relu(lin3), keep_prob=dropout_keep_prob), 10)
    return lin4

def inference_convtree(inputs, is_training, batch_norm=False):
    conv = conv_bn_relu('conv', inputs, 8, is_training=is_training, batch_norm=batch_norm,
                        kernel_size=3, wd=1e-4, strides=(2, 2))

    mconv1 = marginal_bn_relu ('mconv1', conv, 8, is_training=is_training, batch_norm=batch_norm,
                            kernel_size=3, wd=1e-4)
    mconv2 = marginal_bn_relu ('mconv2', mconv1, 4, is_training=is_training, batch_norm=batch_norm,
                            kernel_size=3, wd=1e-4)
    mconv3 = marginal_bn_relu ('mconv3', mconv2, 2, is_training=is_training, batch_norm=batch_norm,
                            kernel_size=3, wd=1e-4)
    flattened = tf.reshape(mconv3, (128, 2048))
    lin1 = linear('lin1', flattened, 1024)
    lin2 = linear('lin2', lin1, 10)
    
    return lin2


def inference_1conv_multiscale(inputs, is_training, batch_norm=False):

    conv3x3 = conv_bn_relu('conv3x3', inputs, 4, is_training=is_training,
                           batch_norm=batch_norm, kernel_size=3, wd=1e-4,
                           strides=(2, 2))
    conv5x5 = conv_bn_relu('conv5x5', inputs, 4, is_training=is_training,
                           batch_norm=batch_norm, kernel_size=5, wd=1e-4,
                           strides=(2, 2))
    conv7x7 = conv_bn_relu('conv7x7', inputs, 4, is_training=is_training,
                           batch_norm=batch_norm, kernel_size=7, wd=1e-4,
                           strides=(4, 4))
    conv9x9 = conv_bn_relu('conv9x9', inputs, 4, is_training=is_training,
                           batch_norm=batch_norm, kernel_size=9, wd=1e-4,
                          strides=(4, 4))
    flat3 = tf.reshape(conv3x3, (128, 1024))
    flat5 = tf.reshape(conv5x5, (128, 1024))
    flat7 = tf.reshape(conv7x7, (128, 256))
    flat9 = tf.reshape(conv9x9, (128, 256))

    all_flat = tf.concat(1, [flat3, flat5, flat7, flat9])  # 128 x 2560
    lin1 = linear('lin1', all_flat, 1024)
    #lin2 = linear('lin2', lin1, 1024)
    lin3 = linear('lin3', lin1, 10)

    return lin3
    

"""
inputs = tf.placeholder(tf.float32, [128, 32, 32, 3])
outputs = inference(inputs, is_training=True)
"""
