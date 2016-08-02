"""
Author: angles
Date and time: 27/07/16 - 14:14
"""
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from batch_functions import provide_batch
from haar_network import inference

# Creating batches for training and testing
batch_size = 128
with tf.name_scope('batch_training'):
    batch_images_training_tensor, batch_labels_training_tensor = provide_batch('train', batch_size, training=True)
with tf.name_scope('batch_testing_test'):
    batch_images_test_tensor, batch_labels_test_tensor = provide_batch('test', batch_size, training=False)

# Predicting the labels using our inference function
with tf.variable_scope('inference') as scope:
    logits_training_tensor = inference(batch_images_training_tensor, is_training=True)
    scope.reuse_variables()
    logits_testing_test_tensor = inference(batch_images_test_tensor, is_training=False)

with tf.name_scope('accuracy'):
    top1_test_tensor = tf.nn.in_top_k(logits_testing_test_tensor, batch_labels_test_tensor, 1)

# Computing the loss
with tf.name_scope('compute_loss'):
    batch_labels_tensor = tf.cast(batch_labels_training_tensor, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_training_tensor, batch_labels_tensor)
    loss = tf.reduce_mean(cross_entropy)
    tf.scalar_summary('loss', loss)

nb_train_samples = 50000
nb_batches_per_epoch_train = int(nb_train_samples / batch_size)
global_step = tf.Variable(0, trainable=False)

with tf.name_scope('optimizer'):
    decay_steps = 60 * nb_batches_per_epoch_train
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=decay_steps,
                                               decay_rate=0.2, staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    with tf.control_dependencies([loss]):
        update_variables = optimizer.minimize(loss, global_step)

saver = tf.train.Saver()

# Create a session for running operations in the Graph
######################################################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Initialize the variables.
sess.run(tf.initialize_all_variables())
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# Saving the graph to visualize in Tensorboard
summary_writer = tf.train.SummaryWriter('./logs', sess.graph)
merged_summary_operation = tf.merge_all_summaries()
# The main loop
###############
nb_test_samples = 10000
nb_epochs_train = 5000
nb_batches_per_epoch_test = int(nb_test_samples / batch_size)
nb_training_steps = nb_epochs_train * nb_batches_per_epoch_train
txt_file = open('results.txt', 'w')
txt_file.write('Training a DCN on CIFAR-10\n')
txt_file.close()
try:
    # If we want to continue training a model:
    # saver.restore(sess, "./model.ckpt")
    epoch = 0
    print(datetime.now(), end=' ')
    for step in range(nb_training_steps):
        # Training
        sess.run(update_variables)
        if (step + 1) % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        if (step + 1) % nb_batches_per_epoch_train == 0:
            merged_summary = sess.run(merged_summary_operation)
            summary_writer.add_summary(merged_summary, step)
            txt_file = open('results.txt', 'a')
            epoch += 1
            print(' Epoch %i' % epoch, end='')
            txt_file.write('%s Epoch %i' % (datetime.now(), epoch))
            # Evaluating test set
            accuracy_epoch = 0
            for index_batch in range(nb_batches_per_epoch_test):
                top1_test = sess.run([top1_test_tensor])
                accuracy_epoch += np.sum(top1_test)
            accuracy_epoch /= nb_batches_per_epoch_test * batch_size
            print(' Total test accuracy: %f \n' % accuracy_epoch)
            txt_file.write(' Total test accuracy: %f\n' % accuracy_epoch)
            saver.save(sess, "./model.ckpt")
            txt_file.close()
            print(datetime.now(), end=' ')
except tf.errors.OutOfRangeError:
    print('Something happened with the queue runners')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
