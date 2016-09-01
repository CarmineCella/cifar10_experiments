import tensorflow as tf
import numpy as np
from itertools import product


def haar1d(x, axis, concat_axis=None):
    xshape = tf.shape(x)
    xndim = len(x.get_shape())
    new_shape = tf.concat(0, (xshape[:axis], tf.pack((xshape[axis] // 2,
                            tf.constant(2))), xshape[axis + 1:]))
    perm = tf.concat(0, (tf.pack((axis + 1,)),  tf.range(0, axis + 1),
                         tf.range(axis + 2, xndim + 1)))
    reorganized = tf.transpose(tf.reshape(x, new_shape), perm)
    even, odd = tf.split(0, 2, reorganized)
    diff = (odd - even) / tf.constant(np.sqrt(2.), dtype=tf.float32)
    summ = (odd + even) / tf.constant(np.sqrt(2.), dtype=tf.float32)
    if concat_axis is None:  # if no axis specified, add one at the beginning
        concat_axis = 0
    else:  # the split leaves an artificial first axis that we need to remove
           # Can't just do diff, summ = diff[0], summ[0] because this stupid
           # shit framework needs the full slice
           # so we have to specify the shape and reshape accordingly
        diff, summ = (tf.reshape(diff, tf.shape(diff)[1:]),
                      tf.reshape(summ, tf.shape(summ)[1:]))
    concat = tf.concat(concat_axis, (diff, summ))
    return concat


def haar3_1d(x, axis, stride=1, concat_axis=None, padding=(1, 1)):
#    xshape = tf.shape(x)
    xshape = x.get_shape()
    xndim = len(x.get_shape())

    full_padding = [(0, 0)] * axis + [padding] + [(0, 0)] * (xndim - axis - 1)
    print(full_padding)
    padded_signal = tf.pad(x, paddings=full_padding,
                           mode='CONSTANT')
    pshape = padded_signal.get_shape()
    
    diff_negative_slice = ([slice(None)] * axis + [slice(2, None, stride)] +
                           [slice(None)] * (xndim - axis - 1))
    diff_positive_slice = ([slice(None)] * axis +
                           [slice(None, pshape[axis].value - 2, stride)] +
                           [slice(None)] * (xndim - axis - 1))
    diff = (padded_signal[diff_positive_slice] -
            padded_signal[diff_negative_slice])
    avg_left_slice = ([slice(None)] * axis +
                      [slice(0, pshape[axis].value - 2, stride)] +
                      [slice(None)] * (xndim - axis - 1))
    avg_middle_slice = ([slice(None)] * axis +
                        [slice(1, pshape[axis].value - 1, stride)] +
                        [slice(None)] * (xndim - axis - 1))
    avg_right_slice = ([slice(None)] * axis + [slice(2, None, stride)] +
                       [slice(None)] * (xndim - axis - 1))

    avg = (padded_signal[avg_left_slice] + padded_signal[avg_middle_slice] +
           padded_signal[avg_right_slice])
    return diff, avg, padded_signal
    

def _3x3_haar_filters():
    diff = np.array([1., 0, -1.]).astype('float32')
    avg = np.ones(3, dtype='float32')
    diff_avg = [diff, avg]
    filters = [np.outer(a, b) for a, b in
               product(diff_avg, diff_avg)]
    return np.array(filters).transpose(
        1, 2, 0).reshape(3, 3, 1, 4).astype('float32')

def _3x3_sobel_filters():
    horz = np.array([[1,0,-1], [2,0,-2],[1, 0, 1]]).astype('float32')
    vert = np.array([[1,2,1], [0,0,0],[-1, -2, -1]]).astype('float32')
    sobel = (horz, vert)
    return sobel

def haar3_2d_conv(x, strides=(2, 2), padding='SAME'):
    xshape = x.get_shape()   # pre-evaluation shape
    height, width, channels = [d.value for d in xshape[1:]]
    x_batch_channel_space = tf.transpose(x, (0, 3, 1, 2))
    new_shape = (-1, height, width, 1)
    x_batch_times_channel_reshaped = tf.reshape(x_batch_channel_space,
                                                new_shape)
    
    filters = tf.constant(_3x3_haar_filters())
    conv_strides = (1,) + strides + (1,)
    conv_raw_ = tf.nn.conv2d(x_batch_times_channel_reshaped, filters,
                             strides=conv_strides, padding=padding)
    s0, s1 = strides
    output_shape_ = (-1, channels, height // s0, width // s1, 4)
    conv_reshaped_ = tf.reshape(conv_raw_, output_shape_)
    conv_transposed_ = tf.transpose(conv_reshaped_, (0, 2, 3, 1, 4))
    output_shape = (-1, height // s0, width // s1, channels * 4)
    return tf.reshape(conv_transposed_, output_shape)

def marginal_2d_conv(x, filters, out_channels, strides=(2, 2), padding='SAME'):
    xshape = x.get_shape()   # pre-evaluation shape
    height, width, channels = [d.value for d in xshape[1:]]
    x_batch_channel_space = tf.transpose(x, (0, 3, 1, 2))
    new_shape = (-1, height, width, 1)
    x_batch_times_channel_reshaped = tf.reshape(x_batch_channel_space,
                                                new_shape)
    
    conv_strides = (1,) + strides + (1,)
    conv_raw_ = tf.nn.conv2d(x_batch_times_channel_reshaped, filters,
                             strides=conv_strides, padding=padding)
    s0, s1 = strides
    output_shape_ = (-1, channels, height // s0, width // s1, out_channels)
    conv_reshaped_ = tf.reshape(conv_raw_, output_shape_)
    conv_transposed_ = tf.transpose(conv_reshaped_, (0, 2, 3, 1, out_channels))
    output_shape = (-1, height // s0, width // s1, channels * out_channels)
    return tf.reshape(conv_transposed_, output_shape)

    
def haar(x, axes, concat_axis=None):
    
    if concat_axis is None:
        # then add an axis at the end, recall the function and
        # concatenate on that one
        xshape = tf.shape(x)
        xshape1 = tf.concat(0, (xshape, tf.pack((1,))))
        concat_axis = len(x.get_shape()) 
        return haar(tf.reshape(x, xshape1), axes, concat_axis)
    
    result = x
    for axis in axes:
        result = haar1d(result, axis, concat_axis)

    return result


def nd1dconv(images, fil_matrix, bias=None):
    # because batch_matmul doesn't broadcast, we need tile
    # because tile doesn't do high dim, we are fucked and have to reshape
    # So we may as well use 1x1 convolution
    images_shape = tf.shape(images)
    images_ndim = len(images.get_shape())
    images_processing_shape = tf.concat(0,
        (images_shape[:2], tf.pack((-1, images_shape[images_ndim - 1]))))
    images_reshaped = tf.reshape(images, images_processing_shape)
    filters_shape = tf.concat(0, (tf.pack((1, 1)), tf.shape(fil_matrix)))
    filters_reshaped = tf.reshape(fil_matrix, filters_shape)
    conv_output = tf.nn.conv2d(images_reshaped, filters_reshaped,
                               (1, 1, 1, 1), 'SAME')
    if bias is not None:
        output = tf.nn.bias_add(conv_output, bias)
    else:
        output = conv_output
    output_shape = tf.concat(0, (images_shape[:images_ndim - 1],
                                 tf.pack((tf.shape(fil_matrix)[1], ))))
    
    return tf.reshape(output, output_shape)


def haar_and_1x1_relu(input_tensor, n_output_channels, scope_name,
                      axes=None, concat_axis=None,
                      ndim=None, is_training=None, batch_norm=False,
                      input_shape=None, output_shape=None):

    if ndim is None:
        ndim = len(input_tensor.get_shape())
    if axes is None:
        axes = list(range(1, ndim))
    with tf.variable_scope(scope_name) as scope:
        haar_transformed = haar(input_tensor, axes, concat_axis=concat_axis)
        if concat_axis is None:
            channel_mixer_input_dim = 2 ** len(axes)
        else:
            channel_mixer_input_dim = input_shape[concat_axis] * 2 ** len(axes)
            if concat_axis in axes:
                channel_mixer_input_dim //= 2
            
        print(channel_mixer_input_dim)
        channel_mixer = tf.get_variable(
             'channel_mixer',
             shape=(channel_mixer_input_dim, n_output_channels),
             dtype=tf.float32,
             initializer=tf.contrib.layers.xavier_initializer())
        channel_mixer_bias = tf.get_variable(
            'bias',
            shape=(n_output_channels,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(.1))
        channel_mixed = nd1dconv(haar_transformed, channel_mixer,
                                 bias=channel_mixer_bias)

        relu = tf.nn.relu(channel_mixed)
        if output_shape is None:
            if input_shape is not None:
                if concat_axis is None:
                    output_shape = np.concatenate([input_shape,
                                                   (n_output_channels,)])
                    for axis in axes:
                        output_shape[axis] //= 2
                else:
                    output_shape = np.array(input_shape)
                    for axis in axes:
                        output_shape[axis] //= 2
                    output_shape[-1] = n_output_channels
                print(output_shape)
                relu = tf.reshape(relu, output_shape)
        else:
            relu = tf.reshape(relu, output_shape)
                
        if batch_norm:
            if is_training not in (True, False):
                raise ValueError(
                    'If using batch_normalization, '
                    'is_training needs to be '
                    'set to True or False. Currently {}'.format(is_training))
            output = tf.contrib.layers.batch_norm(relu, is_training=is_training)
        else:
            output = relu                
    return output


def test_haar3_1d(image=None):
    if image is None:
        from skimage.data import coffee
        image = (coffee() / 256.).astype('float32').reshape(1, 400, 600, 3)
    x = tf.placeholder(tf.float32, shape=image.shape)
    diff, avg, p = haar3_1d(x, axis=1)

    sess = tf.Session()
    o1 = sess.run(diff, {x: c})
    o2 = sess.run(avg, {x: c})
    o3 = sess.run(p, {x: c})
    
    return o1, o2, o3


if __name__ == '__main__':

    from skimage.data import coffee

    c = (coffee() / 256.).astype('float32').reshape(1, 400, 600, 3)

    # x = tf.placeholder(tf.float32, shape=(1, 400, 600, 3))
    # h = haar3_2d_conv(x)

    # sess = tf.Session()
    # o = sess.run(h, {x: c})

    o1, o2, o3 = test_haar3_1d()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(o1[0].mean(-1))
    plt.gray()
    plt.subplot(1, 2, 2)
    plt.imshow(o2[0].mean(-1))
    plt.figure()
    plt.imshow(o3[0])
    plt.show()
    
    
    
