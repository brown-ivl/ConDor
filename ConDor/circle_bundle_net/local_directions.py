import tensorflow as tf
import numpy as np
from circle_bundle_net.local_alignment import frame_from_vector

def local_pca(P, weights=None, center=True):
    """
    :param P: local patches
    :param weights: weights
    :return: local frames and
    """

    # build the covariance matrices
    if center:
        c = tf.reduce_mean(P, axis=2, keepdims=True)
        P = tf.subtract(P, c)

    M = tf.multiply(tf.expand_dims(P, axis=-2), tf.expand_dims(P, axis=-1))
    if weights is not None:
        M = tf.multiply(tf.reshape(weights, list(weights.shape) + [1, 1]))
        M = tf.reduce_sum(M, axis=2, keepdims=False)
    else:
        M = tf.reduce_mean(M, axis=2, keepdims=False)

    e, v = tf.linalg.eigh(M)
    return e, v



def local_direction(patches, points, weights=None, center_patches=True, center=None, cos_eps=0.25, ratio_line=2, ratio_plate=2):
    e, v = local_pca(patches, weights=weights, center=center_patches)

    # optional take singular values as they scale linearly
    e = tf.sqrt(tf.maximum(e, 1e-12))

    e1 = e[..., 0]
    e2 = e[..., 1]
    e3 = e[..., 2]

    v1 = v[..., 0]
    v2 = v[..., 1]
    v3 = v[..., 2]

    # test if we are on a line
    line = tf.greater(e3, ratio_line*e2)
    # plate is not a line and e1 / e2 is small
    plate = tf.logical_and(tf.logical_not(line), tf.greater(e2, ratio_plate*e1))
    # mask blobs
    # blob_mask = tf.logical_and(tf.logical_or(line, plate), tf.greater(e3, 1e-6))
    blob_mask = tf.logical_not(tf.logical_or(line, plate))

    if center is None:
        center = tf.reduce_mean(points, axis=1, keepdims=True)

    r = radial_field(tf.subtract(points, center))

    """
    n = tf.where(tf.expand_dims(line, axis=-1), v3, v1)

    # n = tf.where(tf.expand_dims(blob_mask, axis=-1), r, n)

    c = tf.reduce_sum(tf.multiply(n, r), axis=-1, keepdims=True)
    sgn = tf.sign(c)
    n = tf.multiply(sgn, n)
    oriented = tf.abs(c) > cos_eps
    n = tf.where(oriented, n, r)
    """




    c1 = tf.reduce_sum(tf.multiply(v1, r), axis=-1, keepdims=True)
    c3 = tf.reduce_sum(tf.multiply(v3, r), axis=-1, keepdims=True)
    sgn1 = tf.sign(c1)
    sgn3 = tf.sign(c3)
    v1 = tf.multiply(sgn1, v1)
    v3 = tf.multiply(sgn3, v3)
    n = tf.where(tf.expand_dims(line, axis=-1), v3, v1)
    # c = tf.reduce_sum(tf.multiply(n, r), axis=-1, keepdims=True)
    # n = tf.where(tf.logical_or(tf.expand_dims(blob_mask, axis=-1), tf.abs(c) < cos_eps), r, n)
    # n = tf.where(tf.expand_dims(blob_mask, axis=-1), r, n)
    # return n, line, plate, blob_mask


    return tf.stop_gradient(n), line, plate, blob_mask






def local_direction_(P, weights=None, center=True):
    e, v = local_pca(P, weights=weights, center=center)

    # optional take singular values as they scale linearly
    e = tf.sqrt(tf.maximum(e, 1e-12))

    e1 = e[..., 0]
    e2 = e[..., 1]
    e3 = e[..., 2]

    # test if we are on a line
    line = tf.greater(e3, 2.*e2)
    # plate is not a line and e1 / e2 is small
    plate = tf.logical_and(tf.logical_not(line), tf.greater(e2, 2.*e1))
    # mask blobs
    blob_mask = tf.logical_and(tf.logical_or(line, plate), tf.greater(e3, 1e-6))

    # extract directions
    # line idx is v3 #plate idx is v1
    idx = 2*tf.cast(line, dtype=tf.int32)
    batch_idx = tf.expand_dims(tf.range(P.shape[0]), axis=-1)
    batch_idx = tf.tile(batch_idx, (1, P.shape[1]))
    idx = tf.stack([batch_idx, idx], axis=-1)

    N = tf.gather_nd(v, idx)
    # make it a bidirectional field
    # N = tf.stack([N, -N], axis=-2)

    return N, line, plate, blob_mask

def local_bidirection(P, weights=None, center=True):
    N, line, plate, blob_mask = local_direction(P, weights=weights, center=center)
    N = tf.stack([N, -N], axis=-2)
    return N, line, plate, blob_mask

def center_orientation(x, v, center=True):
    """
    :param x: a collection of points
    :param v: a vector field over x
    :return: the vector field w s.t. w_i = sign(<x_i, v_i>)v_i
    """

    if center:
        c = tf.reduce_mean(x, axis=1, keepdims=True)
        x = tf.subtract(x, c)

    c = tf.reduce_sum(tf.multiply(x, v), axis=-1, keepdims=True)
    sgn = tf.sign(c)
    return tf.multiply(sgn, v)

def orient(u, v):
    """
    :param u: a vector field
    :param v: a vector field
    :return: the vector field w s.t. w_i = sign(<u_i, v_i>)u_i
    """
    c = tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=True)
    sgn = tf.sign(c)
    return tf.multiply(sgn, u)

def radial_field(x):
    c = tf.reduce_mean(x, axis=1, keepdims=True)
    x = tf.subtract(x, c)
    v = tf.linalg.l2_normalize(x, axis=-1)
    return v

def radial_frames(x):
    return frame_from_vector(radial_field(x))

"""
def scaled_distance_patches(P, patch_idx, v, line, center=True):
    if center:
        # center patches
        c = tf.reduce_mean(P, axis=2, keepdims=True)
        P = tf.subtract(P, c)



    line_proj = tf.einsum('bvi,bvpi->bvp', v[..., 2], P)
    plate_proj = P - tf.multiply(tf.expand_dims(line_proj, axis=1), tf.expand_dims(v[..., 2], axis=2))
"""
