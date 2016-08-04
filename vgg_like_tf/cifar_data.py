import os
import tensorflow as tf

cifar_batches_bin_dir = "/tmp/cifar10_data/cifar-10-batches-bin"

def get_file_queue(test=False, cifar_batches_bin_dir=cifar_batches_bin_dir):

    if test:
        filenames = [os.path.join(cifar_batches_bin_dir,
                                  'test_batch.bin')]
    else:
        filenames = [os.path.join(cifar_batches_bin_dir,
                                  'data_batch_{:d}.bin'.format(i))
                     for i in range(1, 6)]
    return tf.train.string_input_producer(filenames, shuffle=True)

