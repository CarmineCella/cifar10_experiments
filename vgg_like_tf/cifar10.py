# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 100,
#                             """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd,
                                initializer='truncated_normal'):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """

  if initializer == 'truncated_normal':
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  
  var = _variable_on_cpu(name, shape, initializer=initializer)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd)#, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def _conv_bn_relu(inp, n_output_channels, scope_name=None):
  with tf.variable_scope(scope_name) as scope:
    n_input_channels = inp.get_shape()[3]
    kernel = _variable_with_weight_decay(
        'weights', shape=(3, 3, n_input_channels, n_output_channels),
        stddev=1e-4, wd=0.0,
        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(inp, kernel, (1, 1, 1, 1), padding='SAME')
    biases = _variable_on_cpu('biases', (n_output_channels,),
                              tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    bn = tf.contrib.layers.batch_norm(bias)
    conv1 = tf.nn.relu(bn, name=scope.name if scope is not None else None)
    return conv1
    


def inference(images, dropout_keep_prob=.4):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # ConvBNReLU(3,64):add(nn.Dropout(0.3))
  # ConvBNReLU(64,64)
  # vgg:add(MaxPooling(2,2,2,2):ceil())
  conv1 = _conv_bn_relu(images, 64, scope_name='conv1')
  drop1 = tf.nn.dropout(conv1, dropout_keep_prob);
  conv2 = _conv_bn_relu(drop1, 64, scope_name='conv2')
  # in torch it says "ceiling mode". No idea wtf that means
  pool1 = tf.nn.max_pool(conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                         padding='SAME')

  # ConvBNReLU(64,128):add(nn.Dropout(0.4))
  # ConvBNReLU(128,128)
  # vgg:add(MaxPooling(2,2,2,2):ceil())  
  conv3 = _conv_bn_relu(pool1, 128, scope_name='conv3')
  drop2 = tf.nn.dropout(conv3, dropout_keep_prob);
  conv4 = _conv_bn_relu(drop2, 128, scope_name='conv4')
  pool2 = tf.nn.max_pool(conv4, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                         padding='SAME')

  # ConvBNReLU(128,256):add(nn.Dropout(0.4))
  # ConvBNReLU(256,256):add(nn.Dropout(0.4))
  # ConvBNReLU(256,256)
  # vgg:add(MaxPooling(2,2,2,2):ceil())
  conv5 = _conv_bn_relu(pool2, 256, scope_name='conv5')
  drop3 = tf.nn.dropout(conv5, dropout_keep_prob);
  conv6 = _conv_bn_relu(drop3, 256, scope_name='conv6')
  drop4 = tf.nn.dropout(conv6, dropout_keep_prob)
  conv7 = _conv_bn_relu(drop4, 256, scope_name='conv7')
  pool3 = tf.nn.max_pool(conv7, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                         padding='SAME')
  
  # ConvBNReLU(256,512):add(nn.Dropout(0.4))
  # ConvBNReLU(512,512):add(nn.Dropout(0.4))
  # ConvBNReLU(512,512)
  # vgg:add(MaxPooling(2,2,2,2):ceil())
  conv8 = _conv_bn_relu(pool3, 512, scope_name='conv8')
  drop5 = tf.nn.dropout(conv8, dropout_keep_prob);
  conv9 = _conv_bn_relu(drop5, 512, scope_name='conv9')
  drop6 = tf.nn.dropout(conv9, dropout_keep_prob)
  conv10 = _conv_bn_relu(drop6, 512, scope_name='conv10')
  pool4 = tf.nn.max_pool(conv10, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                         padding='SAME')

  # # ConvBNReLU(512,512):add(nn.Dropout(0.4))
  # # ConvBNReLU(512,512):add(nn.Dropout(0.4))
  # # ConvBNReLU(512,512)
  # # vgg:add(MaxPooling(2,2,2,2):ceil())
  # # vgg:add(nn.View(512))
  # conv11 = _conv_bn_relu(pool4, 512, scope_name='conv11')
  # drop7 = tf.nn.dropout(conv11, dropout_keep_prob);
  # conv12 = _conv_bn_relu(drop7, 512, scope_name='conv12')
  # drop8 = tf.nn.dropout(conv12, dropout_keep_prob)
  # conv13 = _conv_bn_relu(drop8, 512, scope_name='conv13')
  # pool5 = tf.nn.max_pool(conv13, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
  #                        padding='SAME')
  flattened = tf.reshape(pool4, (-1, 2048))
  drop7 = tf.nn.dropout(flattened, dropout_keep_prob)
  W0 = _variable_with_weight_decay('w0', (2048, 1024), 1e-4, wd=None,
                    initializer=tf.contrib.layers.xavier_initializer())
  b0 = _variable_on_cpu('b0', (1024,),
                         tf.constant_initializer(0.0))
  fully0 = tf.matmul(drop7, W0) + b0
  relu0 = tf.nn.relu(fully0)
  W1 = _variable_with_weight_decay('w1', (1024, 512), 1e-4, wd=None,
                    initializer=tf.contrib.layers.xavier_initializer())
  b1 =  _variable_on_cpu('b1', (512,),
                              tf.constant_initializer(0.0))
  fully1 = tf.matmul(relu0, W1) + b1
  bn1 = tf.contrib.layers.batch_norm(fully1)
  relu1 = tf.nn.relu(bn1)
  drop8 = tf.nn.dropout(relu1, dropout_keep_prob);
  W2 = _variable_with_weight_decay('w2', (512, 10), 1e-4, wd=None,
                    initializer=tf.contrib.layers.xavier_initializer())
  b2 =  _variable_on_cpu('b2', (10,),
                              tf.constant_initializer(0.0))
  fully2 = tf.matmul(drop8, W2) + b2
  return fully2

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
