"""Do a convolution and then flatten it out spatially"""

import tensorflow as tf
import math


def cortex_conv(inp, filters, n_out_w=None, n_out_h=None, 
        strides=(1, 1, 1, 1), padding='SAME'):
    """performs a convolution with filters but then rearranges the output
    channels spatially."""


    n_out = filters.get_shape()[3].value
    if n_out is None and (n_out_w is None or n_out_h is None):
        raise Exception("Filter shape not inferrable from filter tensor "
                "and output shape not inferrable from n_out_w and n_out_h.")
    elif n_out is None:
        n_out = n_out_w * n_out_h

    if n_out_h is None:
        if n_out_w is None:
            sqrt = int(math.sqrt(n_out))
            n_out_w = sqrt
        n_out_h = n_out // n_out_w
    else:
        if n_out_w is None:
            n_out_w = n_out // n_out_h

    conv_raw = tf.nn.conv2d(inp, filters, strides=strides, padding=padding)
    shp = [s.value for s in conv_raw.get_shape()]
    reshaped = tf.reshape(conv_raw[:, :, :, :n_out_w * n_out_h],
            (shp[0], shp[1], shp[2], n_out_h, n_out_w))
    transposed = tf.transpose(reshaped, (0, 1, 3, 2, 4))
    output = tf.reshape(transposed, (shp[0], shp[1] * n_out_h, shp[2] * n_out_w,
                                    1))
    return output




if __name__ == "__main__":

    from skimage.data import coffee
    import numpy as np
    c = coffee().astype('float32') / 256.

    conv_kernel = tf.get_variable('kernel', dtype=tf.float32,
            shape=(5, 5, 3, 64),
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

    x = tf.placeholder(tf.float32, shape=(1,) + c.shape)

    convd = cortex_conv(x, conv_kernel)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    out = sess.run(convd, {x: c[np.newaxis]})





    
    
