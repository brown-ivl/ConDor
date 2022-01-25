import tensorflow as tf
import h5py
import os, sys
import numpy as np
sys.path.append("../")
from utils.losses import chamfer_distance_l2_batch, l2_distance_batch, one_way_chamfer_distance_l2_batch
from pooling import kdtree_indexing_, kdtree_indexing
distance_metric = chamfer_distance_l2_batch
# distance_metric = l2_distance_batch

from utils.data_prep_utils import compute_centroids, tf_random_rotate, tf_center

def orient(r):
    """
    shape = list(r.shape)
    shape = shape[:-2]
    _, u, v = tf.linalg.svd(r)

    R = tf.einsum('bij,bkj->bik', u, v)



    s = tf.stack([tf.ones(shape), tf.ones(shape), tf.sign(tf.linalg.det(R))], axis=-1)
    # u = tf.einsum('bj,bij->bij', s, u)
    u = tf.multiply(tf.expand_dims(s, axis=-1), u)
    # v = tf.multiply(tf.expand_dims(s, axis=1), v)
    R = tf.einsum('bij,bkj->bik', u, v)
    """

    return r

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def save_h5(h5_filename, data, normals=None, subsamplings_idx=None, part_label=None,
            class_label=None, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)

    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    if normals is not None:
        h5_fout.create_dataset(
            'normal', data=normals,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)

    if subsamplings_idx is not None:
        for i in range(len(subsamplings_idx)):
            name = 'sub_idx_' + str(subsamplings_idx[i].shape[1])
            h5_fout.create_dataset(
                name, data=subsamplings_idx[i],
                compression='gzip', compression_opts=1,
                dtype='int32')

    if part_label is not None:
        h5_fout.create_dataset(
            'pid', data=part_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)

    if class_label is not None:
        h5_fout.create_dataset(
            'label', data=class_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def tf_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
    t = tf.random.uniform(shape + [3], minval=0., maxval=1.)
    c1 = tf.cos(2 * np.pi * t[:, 0])
    s1 = tf.sin(2 * np.pi * t[:, 0])

    c2 = tf.cos(2 * np.pi * t[:, 1])
    s2 = tf.sin(2 * np.pi * t[:, 1])

    z = tf.zeros(shape)
    o = tf.ones(shape)

    R = tf.stack([c1, s1, z, -s1, c1, z, z, z, o], axis=-1)
    R = tf.reshape(R, shape + [3, 3])

    v1 = tf.sqrt(t[:, -1])
    v3 = tf.sqrt(1-t[:, -1])
    v = tf.stack([c2 * v1, s2 * v1, v3], axis=-1)
    H = tf.tile(tf.expand_dims(tf.eye(3), axis=0), (batch_size, 1, 1)) - 2.* tf.einsum('bi,bj->bij', v, v)
    M = -tf.einsum('bij,bjk->bik', H, R)
    return M


def batch_of_frames(n_frames, filename, path):

    I = tf.expand_dims(tf.eye(3), axis=0)
    R = tf_random_rotation(n_frames - 1)

    R = tf.concat([I, R], axis=0)
    print("R shape")
    print(R.shape)
    print(R)

    h5_fout = h5py.File(os.path.join(path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


# batch_of_frames(n_frames=128, filename="rotations.h5", path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024")



AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]

def save_rotation(h5_filename, src_path, tar_path, rots_per_shape=512, batch_size=512):
    filename = os.path.join(src_path, h5_filename)
    f = h5py.File(filename)
    print(filename)
    data = f['data'][:]
    num_shapes = data.shape[0]
    num_batches = num_shapes // batch_size
    residual = num_shapes % batch_size
    R = []

    """
    if num_batches == 0:
        batch = tf_random_rotation(rots_per_shape * num_shapes)
        batch = tf.reshape(batch, (-1, rots_per_shape, 3, 3))
        R.append(np.asarray(batch, dtype=np.float))
    """

    for i in range(num_batches):
        a = i*batch_size
        b = min((i+1)*batch_size, num_shapes)
        if a < b:
            batch = tf_random_rotation((b - a)*rots_per_shape)
            batch = tf.reshape(batch, (-1, rots_per_shape, 3, 3))
            batch = np.asarray(batch, dtype=np.float)
            R.append(batch)


    if residual > 0:
        batch = tf_random_rotation(residual * rots_per_shape)
        batch = tf.reshape(batch, (-1, rots_per_shape, 3, 3))
        batch = np.asarray(batch, dtype=np.float)
        R.append(batch)



    # R = tf.concat(R, axis=0)
    R = np.concatenate(R, axis=0)
    print(data.shape)
    print(R.shape)

    # R = np.asarray(R, dtype=np.float)


    h5_fout = h5py.File(os.path.join(tar_path, h5_filename), 'w')

    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


"""
AtlasNetPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024"

for name in AtlasNetClasses:
    save_rotation(name, os.path.join(AtlasNetPath, 'valid'), os.path.join(AtlasNetPath, 'rotations_valid'))
    save_rotation(name, os.path.join(AtlasNetPath, 'train'), os.path.join(AtlasNetPath, 'rotations_train'))


exit(666)
"""

def mean(x, batch_size=512):
    num_shapes = x.shape[0]
    num_batches = num_shapes // batch_size
    remainder = num_shapes // batch_size
    m = []
    k = 0.


    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            k += float(b - a)
            batch = x[a:b, ...]
            m.append(tf.reduce_sum(batch, axis=0, keepdims=True))

    if remainder > 0:
        a = num_batches * batch_size
        b = num_shapes
        if a < b:
            k += float(b - a)
            batch = x[a:b, ...]
            m.append(tf.reduce_sum(batch, axis=0, keepdims=True))

    m = tf.concat(m, axis=0)
    m = tf.reduce_sum(m, axis=0, keepdims=False)
    m /= k
    return m

def var(x, batch_size=512):
    num_shapes = x.shape[0]
    num_batches = num_shapes // batch_size
    remainder = num_shapes // batch_size
    v = []
    k = 0.
    m = tf.expand_dims(mean(x, batch_size=512), axis=0)


    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            k += float(b - a)
            xi = x[a:b, ...]

            vi = tf.subtract(xi, m)
            vi = vi * vi
            vi = tf.reduce_sum(vi)
            v.append(vi)

    if remainder > 0:
        a = num_batches * batch_size
        b = num_shapes
        if a < b:
            k += float(b - a)
            xi = x[a:b, ...]
            vi = tf.subtract(xi, m)
            vi = vi * vi
            vi = tf.reduce_sum(vi)
            v.append(vi)

    v = tf.stack(v, axis=0)
    v = tf.reduce_sum(v)
    v /= k
    return v

def std(x, batch_size=512):
    return tf.sqrt(var(x, batch_size))

def sq_dist_mat(x, y):
    r0 = tf.multiply(x, x)
    r0 = tf.reduce_sum(r0, axis=2, keepdims=True)

    r1 = tf.multiply(y, y)
    r1 = tf.reduce_sum(r1, axis=2, keepdims=True)
    r1 = tf.transpose(r1, [0, 2, 1])

    sq_distance_mat = r0 - 2. * tf.matmul(x, tf.transpose(y, [0, 2, 1])) + r1
    return sq_distance_mat


def var_(x, axis_mean=0, axis_norm=1):
    mean = tf.reduce_mean(x, axis=axis_mean, keepdims=True)
    y = tf.subtract(x, mean)
    yn = tf.reduce_sum(y * y, axis=axis_norm, keepdims=False)
    yn = tf.reduce_mean(yn, axis=axis_mean, keepdims=False)
    return yn, mean

def std_(x, axis_mean=0, axis_norm=1):
    yn, mean = var_(x, axis_mean=axis_mean, axis_norm=axis_norm)
    return tf.sqrt(yn), mean


def pca_align(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    centred_x = tf.subtract(x, c)
    covar_mat = tf.reduce_mean(tf.einsum('bvi,bvj->bvij', centred_x, centred_x), axis=1, keepdims=False)
    _, v = tf.linalg.eigh(covar_mat)

    x = tf.einsum('bij,bvi->bvj', v, centred_x)
    return x

def normalize(x):
    s, m = std_(x, axis_mean=1, axis_norm=-1)
    x = tf.divide(tf.subtract(x, m), s)
    return x


def orth_procrustes(x, y):
    x = normalize(x)
    y = normalize(y)
    xty = tf.einsum('bvi,bvj->bij', y, x)
    s, u, v = tf.linalg.svd(xty)
    r = tf.einsum('bij,bkj->bik', u, v)
    return r


def extend_(x, batch_size):
    last_batch = x.shape[0] % batch_size
    if last_batch > 0:
        X_append = []
        for i in range(batch_size - last_batch):
            X_append.append(x[i, ...])
        X_append = tf.stack(X_append, axis=0)
        y = tf.concat([x, X_append], axis=0)
    else:
        y = x
    return y

"""
def class_consistency_frames(model, x, r, batch_size=32):
    num_batches = x.shape[0] // batch_size
    num_shapes = x.shape[0]
    x = extend_(x, batch_size)
    R = []
    for i in range(num_batches):
        a = i * batch_size
        b = (i + 1) * batch_size
        Ri = []
        for j in range(R.shape[1]):
            xi = x[a:b, ...]
            rj = r[:, j, ...]
            xij = tf.einsum("bij,bvj->bvi", rj, xi)
            yij = model(xij)
            rij = orth_procrustes(xi, yij)
            Ri.append(np.asarray(rij, dtype=np.float32))
        Ri = np.stack(Ri, axis=1)
        R.append(Ri)
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]
    return R
"""
def xyz2yzx(x):
    return tf.stack([x[..., 1], x[..., 2], x[..., 0]], axis=-1)

def yzx2xyz(x):
    return tf.stack([x[..., 2], x[..., 0], x[..., 1]], axis=-1)

def yzx2xyzConj(R):
    R = yzx2xyz(tf.linalg.matrix_transpose(R))
    R = tf.linalg.matrix_transpose(R)
    # return xyz2yzx(R)
    return R

def partial_shapes(x, v):
    """
    num_points = x.shape[1]
    num_points_partial = num_points // 2
    batch_size = x.shape[0]
    # v = tf_random_dir(batch_size)
    if len(v.shape) == 2:
        vx = tf.einsum('bj,bpj->bp', v, x)
    else:
        vx = tf.einsum('j,bpj->bp', v, x)
    x_partial_idx = tf.argsort(vx, axis=-1)
    x_partial_idx = x_partial_idx[:, :num_points_partial]
    batch_idx = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_idx = tf.tile(batch_idx, (1, num_points_partial))
    x_partial_idx = tf.stack([batch_idx, x_partial_idx], axis=-1)
    x_partial = tf.gather_nd(x, x_partial_idx)
    x_partial, kd_idx, kd_idx_inv = kdtree_indexing_(x_partial)

    x_partial = tf_center(x_partial, return_center=False)
    """
    # return x_partial, x_partial_idx, kd_idx_inv
    return x, None, None

def class_consistency_frames(r, r_can, batch_size=32):
    num_batches = r.shape[0] // batch_size
    num_shapes = r.shape[0]
    num_rots = min(r.shape[1], r_can.shape[1])
    r = r[:, :num_rots, ...]
    r_can = r_can[:, :num_rots, ...]
    # r_can = yzx2xyzConj(r_can)
    r = extend_(r, batch_size)
    r_can = extend_(r_can, batch_size)



    R = []
    for i in range(num_batches):
        a = i * batch_size
        b = (i + 1) * batch_size
        rj = r[a:b, ...]
        r_can_j = r_can[a:b, ...]
        # Ri = tf.matmul(r_can_j, rj, transpose_a=True)
        #  Ri = tf.matmul(r_can_j, rj, transpose_b=True)
        # Ri = tf.matmul(r_can_j, r_can_j, transpose_b=True)
        Ri = r_can_j
        Ri = orient(Ri)
        # Ri = tf.matmul(r_can_j, rj)
        # Ri = np.stack(np.asarray(Ri, dtype=np.float32), axis=1)
        R.append(np.asarray(Ri, dtype=np.float32))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]
    # print(R)
    return R
"""
def class_consistency_metric_(r_input, r_can, batch_size=32):
    R = class_consistency_frames(r_input, r_can, batch_size=batch_size)
    R = tf.reshape(R, (-1, 9))
    s, m = std_(R)
    # return float(std(R))
    return float(s)

def class_consistency_metric(filename, r_input_path, r_can_path, batch_size=32):
    fr_input = h5py.File(os.path.join(r_input_path, filename), 'r')
    r_input = fr_input['data'][:]
    fr_input.close()
    fr_can = h5py.File(os.path.join(r_can_path, filename), 'r')
    r_can = fr_can['data'][:]
    fr_can.close()
    return class_consistency_metric_(r_input, r_can, batch_size=batch_size)
"""



def class_consistency_metric_(x, r_input, d_input, r_can, r_partial_can, idx=None, batch_size=32):
    num_rots = min(r_input.shape[1], r_can.shape[0])
    n_shapes = x.shape[0]
    if idx is None:
        idx = tf.random.shuffle(tf.range(n_shapes))

    r_can_0 = r_can
    r_can_1 = tf.gather(r_partial_can, idx, axis=0)
    x = extend_(x, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)
    num_batches = x.shape[0] // batch_size
    D = []

    for j in range(num_rots):
        rj = r_input[j, ...]
        dj = d_input[j, ...]
        d = []
        for i in range(num_batches):
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            xi = x[i * batch_size:(i + 1) * batch_size, ...]
            xij_partial, _, _ = partial_shapes(xi, dj)
            # xij_partial = tf_center(xij_partial, return_center=False)
            xij = tf.einsum("ij,bvj->bvi", rj, xij_partial)
            y0i = tf.einsum("bij,bvj->bvi", orient(r_can_0_ij), xij)
            y1i = tf.einsum("bij,bvj->bvi", orient(r_can_1_ij), xij)
            d.append(np.asarray(distance_metric(y0i, y1i), dtype=np.float))
        d = np.concatenate(d, axis=0)
        d = d[:n_shapes, ...]
        D.append(np.mean(d))
    D = np.stack(D, axis=0)
    D = np.mean(D)
    return float(D)

def loadh5(path):
    fx_input = h5py.File(path, 'r')
    x = fx_input['data'][:]
    fx_input.close()
    return x

def class_consistency_metric(filename, x_path, r_input_path, d_input_path, r_can_path, r_partial_can_path,
                             shapes_idx_path=None, batch_size=32, n_iter=10):
    x = loadh5(os.path.join(x_path, filename))
    r_can = loadh5(os.path.join(r_can_path, filename))
    r_input = loadh5(r_input_path)
    d_input = loadh5(d_input_path)
    r_partial_can = loadh5(os.path.join(r_partial_can_path, filename))


    m = 0.

    if shapes_idx_path is not None:
        idx = loadh5(os.path.join(shapes_idx_path, filename))
        # idx = tf.convert_to_tensor(idx, dtype=tf.int64)
        n_iter = min(n_iter, idx.shape[0])
        for i in range(n_iter):
            m += class_consistency_metric_(x, r_input, d_input, r_can, r_partial_can,
                                           tf.convert_to_tensor(idx[i, ...], dtype=tf.int64), batch_size)
    else:
        idx = None
        for i in range(n_iter):
            m += class_consistency_metric_(x, r_input, d_input, r_can, r_partial_can, idx, batch_size)
    return m / n_iter

"""
def class_consistency_metric_(x, r_input, r_can, idx0, idx1, batch_size=32):
    n_shapes = x.shape[0]
    idx = tf.random.shuffle(tf.range(x.shape[0]))
    r_can_shuffle = tf.gather(r_can, idx, axis=0)
    x = extend_(x, batch_size)
    r_can = extend_(r_can, batch_size)
    r_can_shuffle = extend_(r_can_shuffle, batch_size)


    num_batches = x.shape[0] // batch_size
    D = []
    for i in range(num_batches):
        r_can_shuffle_i = r_can_shuffle[i*batch_size:(i+1)*batch_size, 0, ...]
        r_can_i = r_can[i * batch_size:(i + 1) * batch_size, 0, ...]
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        yi = tf.einsum("bij,bvj->bvi", r_can_i, xi)
        yi_shuffle = tf.einsum("bij,bvj->bvi", r_can_shuffle_i, xi)
        D.append(np.asarray(distance_metric(yi, yi_shuffle), dtype=np.float))
    D = np.concatenate(D, axis=0)
    D = D[:n_shapes, ...]
    D = np.mean(D)
    return float(D)


def class_consistency_metric_(x, r_can, batch_size=32):
    n_shapes = x.shape[0]
    idx = tf.random.shuffle(tf.range(x.shape[0]))
    r_can_shuffle = tf.gather(r_can, idx, axis=0)
    x = extend_(x, batch_size)
    r_can = extend_(r_can, batch_size)
    r_can_shuffle = extend_(r_can_shuffle, batch_size)


    num_batches = x.shape[0] // batch_size
    D = []
    for i in range(num_batches):
        r_can_shuffle_i = r_can_shuffle[i*batch_size:(i+1)*batch_size, 0, ...]
        r_can_i = r_can[i * batch_size:(i + 1) * batch_size, 0, ...]
        xi = x[i * batch_size:(i + 1) * batch_size, ...]
        yi = tf.einsum("bij,bvj->bvi", r_can_i, xi)
        yi_shuffle = tf.einsum("bij,bvj->bvi", r_can_shuffle_i, xi)
        D.append(np.asarray(distance_metric(yi, yi_shuffle), dtype=np.float))
    D = np.concatenate(D, axis=0)
    D = D[:n_shapes, ...]
    D = np.mean(D)
    return float(D)

def class_consistency_metric(filename, x_path, r_can_path, batch_size=32):
    fx_input = h5py.File(os.path.join(x_path, filename), 'r')
    x = fx_input['data'][:]
    fx_input.close()
    fr_can = h5py.File(os.path.join(r_can_path, filename), 'r')
    r_can = fr_can['data'][:]
    fr_can.close()
    return class_consistency_metric_(x, r_can, batch_size=batch_size)
"""



"""
def class_consistency_metric(R):
    R = tf.reshape(R, (-1, 9))
    return std(R)
"""


def equivariance_metric_(x, r_input, d_input, r_can, r_partial_can, batch_size, idx=None):
    num_shapes = x.shape[0]
    num_rots = min(r_input.shape[1], r_can.shape[0])
    if idx is None:
        idx = tf.random.shuffle(tf.range(num_rots))

    r_can_0 = r_can
    r_can_1 = tf.gather(r_partial_can, idx, axis=1)
    r_input_0 = r_input
    r_input_1 = tf.gather(r_input, idx, axis=0)

    x = extend_(x, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)
    # r_input_0 = extend_(r_input_0, batch_size)
    # r_input_1 = extend_(r_input_1, batch_size)

    num_batches = x.shape[0] // batch_size
    D = []
    for i in range(num_batches):
        d = []
        for j in range(num_rots):
            dj = d_input[j, ...]
            r0j = r_input_0[j, ...]
            r1j = r_input_1[j, ...]
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            xi = x[i * batch_size:(i + 1) * batch_size, ...]
            xij_partial, _, _ = partial_shapes(xi, dj)
            # xij_partial = tf_center(xij_partial, return_center=False)
            x0ij = tf.einsum("ij,bvj->bvi", r0j, xij_partial)
            y0i = tf.einsum("bij,bvj->bvi", orient(r_can_0_ij), x0ij)
            x1ij = tf.einsum("ij,bvj->bvi", r1j, xij_partial)
            y1i = tf.einsum("bij,bvj->bvi", orient(r_can_1_ij), x1ij)
            d.append(np.asarray(distance_metric(y0i, y1i), dtype=np.float))
        d = np.stack(d, axis=1)
        d = np.mean(d, axis=1, keepdims=False)
        D.append(d)
    D = np.concatenate(D, axis=0)
    D = D[:num_shapes, ...]
    D = np.mean(D)
    return float(D)

def equivariance_metric(filename, x_path, r_input_path, d_input_path, r_can_path, r_partial_can_path, batch_size, idx_path=None, n_iter=10):
    x = loadh5(os.path.join(x_path, filename))
    r_input = loadh5(r_input_path)
    d_input = loadh5(d_input_path)
    r_can = loadh5(os.path.join(r_can_path, filename))
    r_partial_can = loadh5(os.path.join(r_partial_can_path, filename))
    m = 0.
    if idx_path is None:
        idx = None
        for i in range(n_iter):
            m += equivariance_metric_(x, r_input, d_input, r_can, r_partial_can, batch_size, idx=idx)
    else:
        idx = loadh5(idx_path)
        # idx = tf.convert_to_tensor(idx, dtype=tf.int64)
        n_iter = min(n_iter, idx.shape[0])
        for i in range(n_iter):
            m += equivariance_metric_(x, r_input, d_input, r_can, r_partial_can, batch_size, idx=tf.convert_to_tensor(idx[i, ...], dtype=tf.int64))
    return m / n_iter


def class_consistency_umetric_(x, r_input, d_input, r_can, r_partial_can, idx_shapes=None, idx_rots=None, batch_size=32):
    num_rots = min(r_input.shape[1], r_can.shape[0])
    n_shapes = x.shape[0]
    if idx_shapes is None:
        idx_shapes = tf.random.shuffle(tf.range(n_shapes))
    if idx_rots is None:
        idx_rots = tf.random.shuffle(tf.range(num_rots))
    else:
        idx_rots = idx_rots[:num_rots, ...]

    r_can_0 = r_can
    r_can_1 = tf.gather(r_partial_can, idx_rots, axis=1)
    r_can_1 = tf.gather(r_can_1, idx_shapes, axis=0)
    r_input_0 = r_input
    r_input_1 = tf.gather(r_input, idx_rots, axis=0)
    x_0 = x
    x_1 = tf.gather(x, idx_shapes, axis=0)

    x_0 = extend_(x_0, batch_size)
    x_1 = extend_(x_1, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)

    num_batches = x.shape[0] // batch_size
    D = []

    for j in range(num_rots):
        r0j = r_input_0[j, ...]
        r1j = r_input_1[j, ...]
        d = []
        for i in range(num_batches):
            dj = d_input[j, ...]
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            x0i = x_0[i * batch_size:(i + 1) * batch_size, ...]
            x1i = x_1[i * batch_size:(i + 1) * batch_size, ...]

            x0ij_partial, _, _ = partial_shapes(x0i, dj)
            x1ij_partial, _, _ = partial_shapes(x1i, dj)

            x0ij = tf.einsum("ij,bvj->bvi", r0j, x0ij_partial)
            x1ij = tf.einsum("ij,bvj->bvi", r1j, x1ij_partial)
            y0i = tf.einsum("bij,bvj->bvi", orient(r_can_0_ij), x0ij)
            y1i = tf.einsum("bij,bvj->bvi", orient(r_can_1_ij), x1ij)
            d.append(np.asarray(distance_metric(y0i, y1i), dtype=np.float))
        d = np.concatenate(d, axis=0)
        d = d[:n_shapes, ...]
        D.append(np.mean(d))
    D = np.stack(D, axis=0)
    D = np.mean(D)
    return float(D)


def class_consistency_umetric(filename, x_path, r_input_path, d_input_path, r_can_path, r_partial_can_path, idx_shapes_path=None, idx_rots_path=None, batch_size=32, n_iter=10):
    x = loadh5(os.path.join(x_path, filename))
    r_can = loadh5(os.path.join(r_can_path, filename))
    r_partial_can = loadh5(os.path.join(r_partial_can_path, filename))
    r_input = loadh5(r_input_path)
    d_input = loadh5(d_input_path)

    if idx_shapes_path is not None:
        idx_shapes = loadh5(os.path.join(idx_shapes_path, filename))
        # idx_shapes = tf.convert_to_tensor(idx_shapes, dtype=tf.int64)
        n_iter = min(n_iter, idx_shapes.shape[0])
    else:
        idx_shapes = None

    if idx_rots_path is not None:
        idx_rots = loadh5(idx_rots_path)
        # idx_rots = tf.convert_to_tensor(idx_rots, dtype=tf.int64)
        n_iter = min(n_iter, idx_rots.shape[0])
    else:
        idx_rots = None

    m = 0.
    for i in range(n_iter):
        ri = None
        si = None
        if idx_rots is not None:
            ri = idx_rots[i, ...]
        if idx_shapes is not None:
            si = idx_shapes[i, ...]

        m += class_consistency_umetric_(x, r_input, d_input, r_can, r_partial_can,
                                        idx_shapes=tf.convert_to_tensor(si, dtype=tf.int64),
                                        idx_rots=tf.convert_to_tensor(ri, dtype=tf.int64),
                                        batch_size=batch_size)

    return m / n_iter


"""
def equivariance_metric_(r_input, r_can, batch_size):



    R = class_consistency_frames(r_input, r_can, batch_size=batch_size)
    R = tf.reshape(R, (R.shape[0], R.shape[1], 9))
    num_shapes = R.shape[0]
    R = extend_(R, batch_size)
    num_batches = R.shape[0] // batch_size
    S = []
    for i in range(num_batches):
        a = i * batch_size
        b = (i + 1) * batch_size
        si, mi = std_(R[a:b, ...], 1, -1)
        S.append(np.asarray(si))
    S = np.concatenate(S, axis=0)
    # S = S[:num_shapes, ...]


    s = 0.

    for i in range(num_shapes):
        s += float(S[i])
    return s / num_shapes

def equivariance_metric(filename, r_input_path, r_can_path, batch_size=32):
    fr_input = h5py.File(os.path.join(r_input_path, filename), 'r')
    r_input = fr_input['data'][:]
    fr_input.close()
    fr_can = h5py.File(os.path.join(r_can_path, filename), 'r')
    r_can = fr_can['data'][:]
    fr_can.close()

    return equivariance_metric_(r_input, r_can, batch_size)

"""

"""
def equivariance_metric(r_input, r_can, batch_size=512, num_rotations=128):
    num_shapes = r_input.shape[0]
    num_batches = num_shapes // batch_size
    num_rotations = min(min(num_rotations, r_input.shape[1]), r_can.shape[1])
    r_input = r_input[:, :num_rotations, ...]
    r_can = r_can[:, :num_rotations, ...]
    I = tf.reshape(tf.eye(3), (1, 1, 3, 3))
    N = []
    k = 0.
    remainder = num_shapes % batch_size


    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            k += float(b - a)
            ri = tf.matmul(r_input[a:b, ...], r_can[a:b, ...], transpose_a=True)
            ri = tf.subtract(ri, I)
            ri = tf.reshape(ri, (ri.shape[0], -1, 9))
            ni = tf.reduce_sum(ri * ri, axis=-1, keepdims=False)
            ni = tf.reduce_mean(ni, axis=1, keepdims=False)
            N.append(ni)

    if remainder > 0:
        a = num_batches * batch_size
        b = num_shapes
        if a < b:
            k += float(b - a)
            ri = tf.matmul(r_input[a:b, ...], r_can[a:b, ...], transpose_a=True)
            ri = tf.subtract(ri, I)
            ri = tf.reshape(ri, (ri.shape[0], -1, 9))
            ni = tf.reduce_sum(ri * ri, axis=-1, keepdims=False)
            ni = tf.reduce_mean(ni, axis=1, keepdims=False)
            N.append(ni)


    N = tf.reduce_sum(N)
    N /= k
    return tf.sqrt(N)

def closest_points(x, y):
    d2 = sq_dist_mat(x, y)
    idx = tf.argmin(d2, axis=-1)
    batch_idx = tf.expand_dims(tf.range(x.shape[0]), axis=1)
    batch_idx = tf.tile(batch_idx, (1, x.shape[1]))
    idx = tf.stack([batch_idx, idx], axis=-1)
    return idx

def icp(x, y, n_iter=5):
    x = normalize(x)
    Rx = x
    y = normalize(y)
    R = None
    for i in range(n_iter):
        idx = closest_points(Rx, y)
        y_ = tf.gather_nd(y, idx)
        R = orth_procrustes(x, y_)
        Rx = tf.einsum('bij,bvj->bvi', R, x)
    return R
"""

def icp_class_consistency_metric(x, batch_size=32, n_shuffles=10, n_iter=5):
    """
    :param x: canonicalized shapes (num_shapes, num_points, 3)
    :param batch_size:
    :param n_shuffles: number of times we shuffle x for self comparison
    :param n_iter: number of icp iterations
    :return:
    """
    b = x.shape[0]
    u_ = b % batch_size
    n = b // batch_size
    var_ = 0.
    m = tf.expand_dims(tf.reshape(tf.eye(3), (9,)), axis=1)
    for j in range(n_shuffles):
        idx = np.random.permutation(x.shape[0])
        y_ = np.take(x, indices=idx, axis=0)
        k = 0.
        varj = 0.
        for i in range(n):
            k += 1.
            r = icp(x[i * batch_size:(i + 1) * batch_size, ...], y_[i * batch_size:(i + 1) * batch_size, ...], n_iter=n_iter)
            r = tf.reshape(r, (r.shape[0], -1))
            r_m = tf.subtract(r, m)
            r_m = r_m * r_m
            rn = tf.reduce_sum(r_m, axis=-1)
            varj += float(tf.reduce_mean(rn))

        if u_ > 0:
            k += u_ / float(batch_size)
            r = icp(x[n * batch_size:, ...], y_[n * batch_size:, ...], n_iter=n_iter)
            r = tf.reshape(r, (r.shape[0], -1))
            r_m = tf.subtract(r, m)
            r_m = r_m * r_m
            rn = tf.reduce_sum(r_m, axis=-1)
            varj += float(tf.reduce_mean(rn))
        varj /= k

    var_ /= float(n_shuffles)
    return np.sqrt(var_)



"""
def consistency_metric__(input_frames, can_frames):
    return 0.

def consistency_metric_(input_frames, can_frames, batch_size=512, frames_per_shape=128):
    num_shapes = data.shape[0]
    num_batches = num_shapes // batch_size
    R = []

    if frames_per_shape is None:
        frames_per_shape = min(input_frames.shape[1], can_frames.shape[1])
    else:
        frames_per_shape = min(frames_per_shape, min(input_frames.shape[1], can_frames.shape[1]))
    input_frames = input_frames[:, frames_per_shape:, ...]
    can_frames = can_frames[:, frames_per_shape:, ...]

    C = []
    if num_batches == 0:
        C.append(consistency_metric__(input_frames, can_frames))

    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            input_batch = input_frames[a:b, ...]
            can_batch = can_frames[a:b, ...]
            R.append(consistency_metric__(input_batch, can_batch))
    C = tf.concat(R, axis=0)
    C = tf.reduce_mean(C)
    return float(C)
"""
def shapes_permutations(filename, src_path, tar_path, num=128):
    x = loadh5(os.path.join(src_path, filename))
    n_shapes = x.shape[0]

    I = []
    for i in range(num):
        idx = tf.random.shuffle(tf.range(n_shapes))
        idx = np.asarray(idx, dtype=np.int)
        I.append(idx)

    idx = np.stack(I, axis=0)

    h5_fout = h5py.File(os.path.join(tar_path, filename), "w")
    h5_fout.create_dataset(
        'data', data=idx,
        compression='gzip', compression_opts=1,
        dtype='uint32')
    h5_fout.close()

def rot_permutations(tar_path, num_rots, num_perms):
    filename = "rotations_permutations.h5"

    I = []
    for i in range(num_perms):
        idx = tf.random.shuffle(tf.range(num_rots))
        idx = np.asarray(idx, dtype=np.int)
        I.append(idx)
    idx = np.stack(I, axis=0)


    h5_fout = h5py.File(os.path.join(tar_path, filename), "w")
    h5_fout.create_dataset(
        'data', data=idx,
        compression='gzip', compression_opts=1,
        dtype='uint32')
    h5_fout.close()

def tf_random_dir(shape):
    if isinstance(shape, int):
        shape = [shape]
    t = tf.random.uniform(shape + [3], minval=-1., maxval=1.)
    return tf.linalg.l2_normalize(t, axis=-1)

def random_directions(tar_path, num_dirs):
    v = tf_random_dir(num_dirs)
    v = np.asarray(v, dtype=np.float)
    filename = "directions.h5"
    h5_fout = h5py.File(os.path.join(tar_path, filename), "w")
    h5_fout.create_dataset(
        'data', data=v,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


if __name__== "__main__":


    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]
    
    
    
    # AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "cellphone.h5", "watercraft.h5"]
    
    # AtlasNetClasses = ["plane.h5"]
    AtlasNetShapesPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/valid"
    AtlasNetRotPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_valid"
    r_input_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations.h5"
    d_input_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/directions.h5"
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full_multicategory"
    partial_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial"
    partial_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial_multicategory"
    
    """
    for f in AtlasNetClasses:
        shapes_permutations(f, AtlasNetShapesPath, "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations")
    
    exit(666)
    """
    
    # rot_permutations("I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024", 128, 128)
    # random_directions("I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024", 128)
    # exit(666)
    """
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full_multicategory"
    
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full_multicategory"
    """
    # full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_consistency_full"
    # full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_consistency_full_multicategory"
    
    
    # ull_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full"
    # full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full_multicategory"
    
    
    # multicategory full shapes
    """
    print("multi category")
    ma = 0.
    mb = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetRotPath, full_multi_pred_path, batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetRotPath, full_multi_pred_path, batch_size=32)
        print("equivariance: ", b)
        mb += b
        k += 1.
    
    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)
    
    
    print("category specific")
    ma = 0.
    mb = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetRotPath, full_pred_path, batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetRotPath, full_pred_path, batch_size=32)
        print("equivariance: ", b)
        mb += b
        k += 1.
    
    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)
    """
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full_multicategory"
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full_multicategory"
    
    
    
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/caca_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/caca_full_multicategory"
    
    partial_pred_path = ""
    partial_multi_pred_path = ""
    
    # full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full"
    # full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full_multicategory"
    
    
    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]
    
    n_iter = 40
    print("multi category")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, d_input_path, full_multi_pred_path, partial_multi_pred_path,
                                 shapes_idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations", batch_size=48, n_iter=n_iter)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, d_input_path, full_multi_pred_path, partial_multi_pred_path,
                                idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5", batch_size=48, n_iter=n_iter)
        print("equivariance: ", b)
        mb += b
        c = class_consistency_umetric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, d_input_path, full_multi_pred_path, partial_multi_pred_path,
                                      idx_shapes_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                      idx_rots_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                      batch_size=48, n_iter=n_iter)
        mc += c
        print("u_consistency: ", c)
    
        k += 1.
    
    print("mean multi class consistency: ", ma / k)
    print("mean multi class equivariance: ", mb / k)
    print("mean multi class uconsistency: ", mc / k)
    
    
    AtlasNetClasses = ["plane.h5", "chair.h5"]
    
    # AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]
    
    
    print("category specific")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                     shapes_idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                     batch_size=48, n_iter=n_iter)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                batch_size=48, n_iter=n_iter)
        print("equivariance: ", b)
        mb += b
        c = class_consistency_umetric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                      idx_shapes_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                      idx_rots_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                      batch_size=48, n_iter=n_iter)
        mc += c
        print("u_consistency: ", c)
        k += 1.
    
    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)
    print("mean multi class uconsistency: ", mc / k)
