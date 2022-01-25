import tensorflow as tf
import numpy as np
from circle_bundle_net.kernels import tf_complex_powers
import math
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D

def complex_circle_bundle_pooling(F, T=None, pool_mode='avg', weights=None,
                                  invert_transport=True, num_samples=16):
    """
    :param F: input signal on patches
    :param T: transfer map
    :param pool_mode: 'avg' or 'max'
    :param weights: Pooling weights
    :param invert_transport: Shall we invert the transport map if it has been computed for the other direction
    :param k: number of samples for max pooling
    :return num_samples: number of samples for max pooling
    """

    if weights is not None:
        weights = tf.cast(weights, tf.complex64)
        weights = tf.expand_dims(weights, axis=0)
        F = tf.multiply(weights, F)

    if invert_transport and T is not None:
        T = tf.math.conj(T)
        
    if T is not None:
        T = tf.expand_dims(T, axis=-1)
        F = tf.multiply(T, F)

    if pool_mode == 'avg':
        if weights is None:
            y = tf.reduce_mean(F, axis=2, keepdims=False)
        else:
            y = tf.reduce_sum(F, axis=2, keepdims=False)
    else:
        c = tf.cos((2. * np.pi / num_samples) * tf.range(num_samples, dtype=tf.float32))
        s = tf.sin((-2. * np.pi / num_samples) * tf.range(num_samples, dtype=tf.float32))
        e = tf_complex_powers(c, s, int(F.shape[-2] / 2.))
        F = tf.einsum('mi,bvpmc->bvpic', e, F)
        y = tf.reduce_max(F, axis=2, keepdims=False)
    return y

def Log2(x):
    return (math.log10(x) / math.log10(2))
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))


def kd_pooling_1d(x, pool_size, pool_mode='avg'):

    assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode == 'max':
        pool = MaxPooling1D(pool_size)
    else:
        pool = AveragePooling1D(pool_size)
    if isinstance(x, list):
        y = []
        for i in range(len(x)):
            x.append(pool(x[i]))
    elif isinstance(x, dict):
        y = dict()
        for l in x:
            if isinstance(l, int):
                y[l] = pool(x[l])
    else:
        y = pool(x)
    return y

def kd_pooling_2d(x, pool_size, pool_mode='avg'):

    assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode == 'max':
        pool = MaxPooling2D((pool_size, 1))
    else:
        pool = AveragePooling2D((pool_size, 1))
    if isinstance(x, list):
        y = []
        for i in range(len(x)):
            x.append(pool(x[i]))
    elif isinstance(x, dict):
        y = dict()
        for l in x:
            if isinstance(l, int):
                y[l] = pool(x[l])
    else:
        y = pool(x)
    return y

def equivariant_kd_pooling(y, pool_size, alpha=1):
    assert(isinstance(y, dict))
    z = dict()
    for l in y:
        if l.isnumeric():
            ynl = tf.reduce_sum(tf.multiply(y[l], y[l]), axis=-2, keepdims=True)
            ynl = tf.exp(alpha*ynl)
            yl = tf.multiply(ynl, y[l])
            if isinstance(pool_size, int):
                ynl = kd_pooling_2d(ynl, pool_size=pool_size, pool_mode='avg')
                yl = kd_pooling_2d(y[l], pool_size=pool_size, pool_mode='avg')
                z[l] = tf.divide(yl, ynl)
            else:
                ynl = tf.reduce_sum(ynl, axis=1, keepdims=False)
                yl = tf.reduce_sum(yl, axis=1, keepdims=False)
                z[l] = tf.divide(yl, ynl)
    return z

def extract_samples_slices(num_points_total, num_points):
    print(num_points_total)
    assert isPowerOfTwo(num_points_total + 1)
    n = int((num_points_total + 1)/2)
    k = []
    for i in range(len(num_points)):
        assert isPowerOfTwo(num_points[i])
        m = num_points[i]
        ki = int(np.log(n / m) / np.log(2) + 0.00001)
        ki = int(n * (2**(ki+1) - 1) / (2**ki) + 0.00001)
        k.append(ki)
    return k
"""
class KdTreePooling(tf.keras.layers.Layer):
    def __init__(self, mode='MAX'):
        super(KdTreePooling, self).__init__()
        self.mode = mode

    def build(self, input_shape):
        super(KdTreePooling, self).build(input_shape)

    def call(self, x):
        return grid_pooling(x[0], x[1], x[2], x[3], x[4], mode='avg')
"""

def diameter(x, axis=-2, keepdims=True):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims) - tf.reduce_min(x, axis=axis, keepdims=keepdims)

def kdtree_indexing(x, depth=None):
    num_points = x.shape[1]
    assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = tf.range(x.shape[0],dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))

    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = tf.argmax(diam, axis=-1, output_type=tf.int32)
        split_idx = tf.tile(split_idx, (1, y.shape[1]))
        # split_idx = tf.tile(split_idx, (1, y.shape[1], 1))
        idx = tf.range(y.shape[0])
        idx = tf.expand_dims(idx, axis=-1)
        idx = tf.tile(idx, (1, y.shape[1]))
        branch_idx = tf.range(y.shape[1])
        branch_idx = tf.expand_dims(branch_idx, axis=0)
        branch_idx = tf.tile(branch_idx, (y.shape[0], 1))
        split_idx = tf.stack([idx, branch_idx, split_idx], axis=-1)
        m = tf.gather_nd(y, split_idx)
        sort_idx = tf.argsort(m, axis=-1)
        sort_idx = tf.stack([idx, sort_idx], axis=-1)
        y = tf.gather_nd(y, sort_idx)
        y = tf.reshape(y, (-1, int(y.shape[1] // 2), 3))

    y = tf.reshape(y, x.shape)
    return y

def kdtree_indexing_(x, depth=None):
    num_points = x.shape[1]
    assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = tf.range(x.shape[0],dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1))
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))

    points_idx = tf.range(num_points)
    points_idx = tf.reshape(points_idx, (1, -1, 1))
    points_idx = tf.tile(points_idx, (x.shape[0], 1, 1))



    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = tf.argmax(diam, axis=-1, output_type=tf.int32)
        split_idx = tf.tile(split_idx, (1, y.shape[1]))
        # split_idx = tf.tile(split_idx, (1, y.shape[1], 1))
        idx = tf.range(y.shape[0])
        idx = tf.expand_dims(idx, axis=-1)
        idx = tf.tile(idx, (1, y.shape[1]))
        branch_idx = tf.range(y.shape[1])
        branch_idx = tf.expand_dims(branch_idx, axis=0)
        branch_idx = tf.tile(branch_idx, (y.shape[0], 1))
        split_idx = tf.stack([idx, branch_idx, split_idx], axis=-1)
        m = tf.gather_nd(y, split_idx)
        sort_idx = tf.argsort(m, axis=-1)
        sort_idx = tf.stack([idx, sort_idx], axis=-1)
        points_idx = tf.gather_nd(points_idx, sort_idx)
        points_idx = tf.reshape(points_idx, (-1, int(y.shape[1] // 2), 1))
        y = tf.gather_nd(y, sort_idx)
        y = tf.reshape(y, (-1, int(y.shape[1] // 2), 3))

    y = tf.reshape(y, x.shape)
    points_idx = tf.reshape(points_idx, (x.shape[0], x.shape[1]))
    points_idx_inv = tf.argsort(points_idx, axis=-1)
    points_idx = tf.stack([batch_idx, points_idx], axis=-1)
    points_idx_inv = tf.stack([batch_idx, points_idx_inv], axis=-1)
    return y, points_idx, points_idx_inv

def aligned_kdtree_indexing(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    x = tf.add(x, c)
    return kdtree_indexing(x)

def aligned_kdtree_indexing_(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    x = tf.add(x, c)
    y, points_idx, points_idx_inv = kdtree_indexing_(x)
    return y, points_idx, points_idx_inv, v




