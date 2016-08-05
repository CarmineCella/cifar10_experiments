from theano_haar import haar
from keras.engine.topology import Layer
from keras.initializations import glorot_uniform
import keras.backend as K
import numpy as np

class HaarLayer(Layer):
    def __init__(self, **kwargs):
        super(HaarLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass
    
    def call(self, x, mask=None):
        axes = list(range(1, x.ndim))
        h = haar(x, axes=axes, concat_axes=None)
        return h.swapaxes(0, 1)

    def get_output_shape_for(self, input_shape):
        input_shape = np.array(input_shape)
        n_samples = input_shape[0]
        signal_shape = np.array(input_shape[1:])
        n_haar_channels = 2 ** len(signal_shape)
        output_shape = (n_samples, n_haar_channels) + tuple(signal_shape // 2)
        return output_shape

class ChannelMixerLayer(Layer):
    def __init__(self, n_output_channels, **kwargs):
        self.n_output_channels = n_output_channels
        super(ChannelMixerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # initial_weight_value = glorot_uniform((input_dim, self.n_output_channels))
        # self.W = K.variable(initial_weight_value)
        self.W = glorot_uniform((input_dim, self.n_output_channels), name='W')
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        x_reshaped = x.reshape(K.concatenate((x.shape[:3], (-1,))), ndim=4)
        conv_kernel = self.W.T.reshape(
            K.concatenate((self.W.T.shape, (1, 1))), ndim=4)
        conv = K.conv2d(x_reshaped, conv_kernel)
        output_shape = K.concatenate(((x.shape[0], self.n_output_channels),
                                      x.shape[2:]))
        return conv.reshape(output_shape, ndim=x.ndim)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_output_channels)+tuple(input_shape[2:])
    
if __name__ == '__main__':
    from skimage.data import coffee
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.layers import BatchNormalization, Activation
    cof = coffee().mean(2).astype('float32')
    
    model = Sequential()
    model.add(HaarLayer(input_shape=(400, 600)))
    model.add(ChannelMixerLayer(8))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    
                  
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    
    r = model.predict(cof[np.newaxis], 1)
              
