import tensorflow as tf
import numpy as np

def fourier_transform(x, l_max):

    y0 = tf.reduce_mean(x, axis=-2, keepdims=True)
    y = dict()
    y['0'] = y0
    if l_max == 0:
        return y

    n = x.shape[-2]
    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)

    c = tf.cos(t)
    s = tf.sin(t)
    e = tf.stack([c, s], axis=-1)

    y_ = tf.einsum('iab,...ic->...abc', e, x)
    y_ = tf.split(y_, l_max, axis=-3)
    y = dict()
    for m in range(l_max):
        y[str(m)] = y_[m]

    return y


def tf_inverse_shp_fourier_transfrom(x, n):
    types = []
    for l in x:
        if l.isnumeric():
            types.append(int(l))
    types.sort()
    l_max = types[-1]

    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)

    c = tf.cos(t)
    s = tf.sin(t)
    o = tf.ones((n, 1))
    e = tf.concat([s, o, c], axis=-2)

    Y = []
    # dims = [2*l+1 for l in types]
    for l in types:
        el = e[..., l_max - l:l_max+l+1, :]
        yl = tf.matmul(el, x[str(l)])
        Y.append(yl)
    y = tf.concat(Y, axis=-1)
    return y, types

def tf_sph_fourier_transform(x, types):

    types_ = types.copy()
    types_.sort()
    l_max = types_[-1]

    n = x.shape[-2]

    t = (2. * np.pi / n) * tf.range(n, dtype=tf.float32)
    k = tf.range(start=1, limit=l_max + 1, dtype=tf.float32)
    t = tf.expand_dims(t, axis=-1)
    k = tf.expand_dims(k, axis=0)
    t = tf.multiply(t, k)

    c = tf.cos(t)
    s = tf.sin(t)
    o = tf.ones((n, 1))
    e = tf.concat([s, o, c], axis=-2)

    dims = [2 * l + 1 for l in types_]
    y_ = tf.split(x, num_or_size_splits=dims, axis=-1)
    y = dict()
    for i in range(len(types)):
        l = types[i]
        el = e[..., l_max - l:l_max + l + 1, :]
        yl = tf.matmul(el, y_[i], transpose_a=True)
        y[str(l)] = yl
    return y

