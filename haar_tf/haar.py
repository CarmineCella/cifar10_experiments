import tensorflow as tf
import numpy as np

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
        channel_mixer_bias = tf.get_variable('bias',
                                             shape=(n_output_channels,),
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(.1))
        channel_mixed = nd1dconv(haar_transformed, channel_mixer, bias=channel_mixer_bias)

        relu = tf.nn.relu(channel_mixed)
        if output_shape is None:
            if input_shape is not None:
                if concat_axis is None:
                    output_shape = np.concatenate([input_shape, (n_output_channels,)])
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
                raise ValueError('If using batch_normalization, is_training needs to'
                                 ' be set to True or False. Currently {}'.format(is_training))
            output = tf.contrib.layers.batch_norm(relu, is_training=is_training)
        else:
            output = relu                
    return output

