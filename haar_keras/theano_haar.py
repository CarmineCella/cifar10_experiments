################################################################################

import theano.tensor as T
import numpy as np

def haar1d(X, axis, concat_axis=None):
    new_shape = T.concatenate([X.shape[:axis],
                               (X.shape[axis]//2, 2),
                                X.shape[axis+1:]])
    axis_perm = (axis+1,)+tuple(range(axis+1))+tuple(range(axis+2,X.ndim+1))
    X_even_odd = X.reshape(new_shape, ndim=X.ndim+1).transpose(axis_perm)
    even,odd = X_even_odd[0], X_even_odd[1]
    diff,avg = (even-odd) / T.sqrt(2), (even+odd) / T.sqrt(2)
    if concat_axis is None:
        return T.as_tensor_variable((diff, avg))
    else:
        return T.concatenate ((diff, avg), axis=concat_axis)

    
def haar (X, axes=None, concat_axes=None):
    if axes is None:
        axes = list(range (X.ndim))
    else:
        axes = list(axes)
    if concat_axes is None:
        concat_axes = (None,) + (0,) * (len(axes) - 1)
        axes = [axes[0]] + [axis + 1 for axis in axes[1:]]
    elif concat_axes == 'same':
        concat_axes = axes
    print(axes)
    result = X
    for axis, concat_axis in zip(axes, concat_axes):
        print(axis, concat_axis)
        result = haar1d(result, axis, concat_axis=concat_axis)

    return result

        
if __name__ == "__main__":
    from skimage.data import coffee
    cof = coffee()
    cof = np.mean (cof, 2).astype('float32')
    
    x = T.fmatrix()
    s = haar(x, concat_axes=(0, 1))

    r = s.eval ({x:cof})
    
